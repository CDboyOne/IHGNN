from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import time

from Models import PpsModel, PpsModelTypes, RawGnn, Srrl, IHGNNLayer
from Dataset import TestSearchLogDataLoader, GraphDataset
from Helpers.Metrics import Metrics
from Helpers.ProcessController import ProcessController
from Helpers.IOHelper import IOHelper
from Helpers.Torches import *
from Helpers.GlobalSettings import Gs, Gsv

def print_network_parameters(module: nn.Module, name_filter: str = None):

    '''工整地输出模型的所有参数（等宽字体下）。'''

    max_name_len = 0
    max_size_len = 0

    for name, parameter in module.named_parameters():
        if name_filter and (name_filter not in name): continue
        size = '(' + ', '.join([str(n) for n in parameter.size()]) + ')'
        max_name_len = (len(name) if len(name) > max_name_len else max_name_len)
        max_size_len = (len(size) if len(size) > max_size_len else max_size_len)

    for name, parameter in module.named_parameters():
        if name_filter and (name_filter not in name): continue
        size = '(' + ', '.join([str(n) for n in parameter.size()]) + ')'
        grad = ('GRAD  ' if parameter.requires_grad else 'NO_GRAD')
        name = '%*s' % (-max_name_len, name)
        size = '%*s' % (-max_size_len, size)
        with th.no_grad():
            IOHelper.LogPrint(f'{name} | size={size} | {grad} '
                + f'| mean={parameter.mean().item():<7.3f} | std={parameter.std().item():<7.3f} '
                + f'| absmean={parameter.abs().mean().item():<7.3f}')


def test_and_get_avg_metrics(
    model: PpsModel, 
    dataset_train: GraphDataset,
    dataloader: TestSearchLogDataLoader,
    get_long_tail_stat: bool = False) -> Tuple[List[Metrics], Metrics, float]:

    '''返回值：各 user 的平均评价指标（包含 None）（如果不统计则为 None）；所有 user 一起的平均评价指标；测试所用时间。'''

    count_valid = 0
    metrics = Metrics()
    start_time = time.time()

    if get_long_tail_stat:
        u_metrics_list: List[List[Metrics]] = [[] for _ in range(dataset_train.user_count)]
        u_metrics: List[Metrics] = []

    with torch.no_grad():
        
        if type(model) in PpsModelTypes:
            model.save_features_for_test()

        for users, queries, items_interacted, flags_interacted, flags_all_1 in dataloader:

            # Predict on all items when using None
            outputs = model(users, queries, None)
            m = Metrics.calculate_on_all_items(
                model_outputs=outputs,
                interacted_items=items_interacted,
                flags=flags_interacted,
                flags_are_all_1=flags_all_1
            )

            if m is not None:
                metrics.add_to_self(m)
                count_valid += 1

                if get_long_tail_stat:
                    u_metrics_list[users[0].item()].append(m)
        
        if get_long_tail_stat:
            for u in range(dataset_train.user_count):
                ms = u_metrics_list[u]
                if len(ms) == 0:
                    u_metrics.append(None)
                else:
                    m0 = Metrics()
                    for m in ms:
                        m0.add_to_self(m)
                    u_metrics.append(m0.divide_and_get_new(len(ms)))
        
        if type(model) in PpsModelTypes:
            model.clear_saved_feature()
        
        # Calculate average metrics
        metrics = metrics.divide_and_get_new(count_valid)
        IOHelper.LogPrint(f'测试完成，共 {time.time()-start_time:<.2f} s，计 {count_valid} 条有效的 search log.')
        IOHelper.LogPrint(
            metrics.to_string(highlight=True), 
            put_time_in_single_line=True
        )
        IOHelper.LogPrint()

        if get_long_tail_stat:
            return u_metrics, metrics, time.time() - start_time
        else:
            return None, metrics, time.time() - start_time


def train_and_get_avg_loss(
    model: PpsModel,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    dataset_train: GraphDataset,
    dataloader_train: DataLoader,
    pc: ProcessController,
    device: th.device) -> Tuple[float, float]:

    '''返回：平均 loss；训练所用时间。'''

    loss_sum = 0.0
    interaction_count = 0
    time_epoch_start = time.time()

    if not isinstance(model, Srrl):

        # 训练
        for batch_index, (users, queries, items, flags, neg_users, neg_queries, neg_items, neg_flags) in enumerate(dataloader_train):
            interaction_count += len(users)

            users   = th.cat([users, neg_users])
            queries = th.cat([queries, neg_queries])
            items   = th.cat([items, neg_items])
            flags   = th.cat([flags, neg_flags]).float()

            outputs = model(users, queries, items)
            loss = loss_function(outputs, flags)
            loss_sum += loss.item()
            loss.backward()
            # 当 batch 被自动缩小时
            # 然后，每过 batch_size_times 个 batch，或遇到最后一个 batch 时，清除梯度
            if Gs.batch_size_times != 1:
                if ((batch_index + 1) % Gs.batch_size_times == 0 or interaction_count == dataset_train.interaction_count):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
        
        # 计算 loss
        avg_loss = loss_sum / (batch_index + 1)
        IOHelper.LogPrint(
            f'[Epoch \033[0;44m{pc.CurrentEpoch:>2d}/{pc.EndEpoch - 1}\033[0m] ' + 
            f'Average loss \033[0;45m{avg_loss:<.4f}\033[0m ' + 
            f'on {interaction_count} interactions in {time.time()-time_epoch_start:<.2f} s ' + 
            f'(remaining {pc.GetRemainingTimeString()}).'
        )

        # 调整学习率
        if Gs.adjust_learning_rate and isinstance(model, RawGnn) and avg_loss < 0.008 and Gs.learning_rate > 0.0004:
            Gs.learning_rate *= 0.98
            for param_group in optimizer.param_groups:
                param_group['lr'] = Gs.learning_rate
            IOHelper.LogPrint(f'学习率调整为：{Gs.learning_rate}')
    else:
        
        # 训练 Knowledge Graph
        sum_step_loss_KG = 0
        if Gs.Srrl.KG_loss:
            model.train()
            srrl_kg_start = time.time()

            for s in range(model.srrl_steps):
                (positive_sample, negative_sample, subsampling_weight, 
                    mode, true_tail_company, true_head_company, true_query_company) = model.train_iterator_KG.next()

                positive_sample    = positive_sample.to(device)
                negative_sample    = negative_sample.to(device)
                subsampling_weight = subsampling_weight.to(device)
                true_tail_company  = true_tail_company.to(device)
                true_head_company  = true_head_company.to(device)
                true_query_company = true_query_company.to(device)

                optimizer.zero_grad()
                negative_score: Tensor = model.trainkg(
                    sample=(positive_sample, negative_sample, true_tail_company, true_head_company, true_query_company), 
                    mode=mode, 
                    positive_mode=False
                )
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)

                positive_score: Tensor = model.trainkg(
                    (positive_sample, true_tail_company, true_head_company, true_query_company), 
                    mode=mode, 
                    positive_mode=True
                )
                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                if Gs.Srrl.uni_weight:
                    positive_sample_loss = - positive_score.mean()
                    negative_sample_loss = - negative_score.mean()
                else:
                    positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                    negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

                loss = (positive_sample_loss + negative_sample_loss) / 2

                if Gs.Srrl.regularization != 0.0:
                    # Use L2 regularization
                    regulariza = Gs.Srrl.regularization * (
                            model.KG.embedding_user.weight.data.norm(p=2) ** 2 +
                            model.KG.embedding_bag_vocabulary.weight.data.norm(p=2) ** 2 +
                            model.KG.embedding_item.weight.data.norm(p=2) ** 2
                    )
                    loss += Gs.Srrl.regularization * regulariza

                sum_step_loss_KG += loss.item()
                loss.backward()
                optimizer.step()

            avg_loss_KG = sum_step_loss_KG / (s + 1)
            IOHelper.LogPrint(
                f'[Epoch KG {pc.CurrentEpoch:<2d}] avg loss  KG->  {avg_loss_KG:<.4f} in {time.time()-srrl_kg_start:<.2f}s' +
                f' (remaining {pc.GetRemainingTimeString()}).'
            )
        
        # 训练 Personalized Search
        model.train()
        sum_step_loss_PS = 0
        srrl_ps_start = time.time()

        for i, (uids, queries, items, labels, neg_uids, neg_queries, neg_items, neg_labels) in enumerate(dataloader_train):

            uids    = th.cat([uids, neg_uids])
            queries = th.cat([queries, neg_queries])
            items   = th.cat([items, neg_items])
            labels  = th.cat([labels, neg_labels]).float()
            
            optimizer.zero_grad()
            score_normalized = model(uids, queries, items)
            loss = loss_function(score_normalized.float(), labels)
            if Gs.Srrl.regularization != 0.0:
                # Use L2 regularization for ComplEx and DistMult
                regulariza= Gs.Srrl.regularization * (
                        model.PS.embedding_user.weight.data.norm(p=2) ** 2 +
                        model.KG.embedding_bag_vocabulary.weight.data.norm(p=2) ** 2 +
                        model.PS.embedding_item.weight.data.norm(p=2) ** 2
                )
                loss += Gs.Srrl.regularization*regulariza

            sum_step_loss_PS += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss_PS = sum_step_loss_PS / (i + 1)
        avg_loss = avg_loss_PS
        IOHelper.LogPrint(
            f'[Epoch PS {pc.CurrentEpoch:<2d}] avg loss {avg_loss_PS:<.4f}  <-PS  in {(time.time() - srrl_ps_start):<.2f}s' +
            f' (remaining {pc.GetRemainingTimeString()}).'
        )
    
    return avg_loss, (time.time() - time_epoch_start)

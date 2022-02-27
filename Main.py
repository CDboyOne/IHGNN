from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable, Callable
import os, time, sys
os.environ['DGLBACKEND'] = 'pytorch'
os.umask(0)
sys.dont_write_bytecode = True

from Models import RawGnn, Srrl, HemPredictionLayer, parse_model_type
from Models import GCNLayer, GATLayer, HGCNLayer, IHGNNLayer, GnnLayer, parse_gnn_layer
from Dataset import GraphDataset, TestSearchLogDataLoader
from SrrlDataset import MetaPaths, SrrlDatasetKG, OneShotIterator
from Helpers.Torches import *
from Helpers.Graph import Pps2DGraph, PpsHyperGraph
from Helpers.Metrics import Metrics, MetricsCollection
from Helpers.ArgsParser import parse_args
from Helpers.IOHelper import IOHelper
from Helpers.ProcessController import ProcessController
from Helpers.GlobalSettings import Gs, Gsv
from Helpers.TrainTestHelper import *

args = parse_args()
Gs.graph_completeness = args.completeness
Gs.long_tail_stat_fn = args.long_tail_filename
Gs.embedding_size = args.embedding_size or Gs.embedding_size

########## Some settings (Other settings are in GlobalSettings.py) ##########

epoch_count                = args.epoch_count or      110
epoch_start_test_position  = args.epoch_start_test or 10
epoch_test_frequency       = args.epoch_test_frequency or epoch_start_test_position
epoch_start_store_position = epoch_count
epoch_store_frequency      = 1000000

dataset_name = 'Cikm/NoCategory/'
dataset_name = 'Cikm/WithCategory5Core/'
dataset_name = 'Amazon/OfficeProducts/Complete/'
dataset_name = 'Amazon/CellPhones/Complete5Core/'
dataset_name = 'Amazon/KindleStore/Complete5Core/'
dataset_name = 'Amazon/VideoGames/Complete/'
dataset_name = 'Amazon/Electronics/Complete5Core/'
dataset_name = 'AlibabaAir/Subset01W5Core/'
dataset_name = 'AlibabaAir/Complete1Core/'
dataset_name = 'AlibabaAir/Complete5Core/'
dataset_name = args.dataset or dataset_name

model_type = Srrl #
model_type = RawGnn #
model_type = parse_model_type[args.model] or model_type

gnn_layer_type = GCNLayer #
gnn_layer_type = GATLayer #
gnn_layer_type = HGCNLayer #
gnn_layer_type = IHGNNLayer #
gnn_layer_type = parse_gnn_layer[args.gnn] or gnn_layer_type

gnn_layer_count = args.gnns or 2
feature_interaction_order = args.feature_order or 3
phase2_attention = False

prediction_layer_type = HemPredictionLayer

device = torch.device('cpu')
device = torch.device('cuda:0')
if bool(args.device):
    device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{args.device}')

#############################################################################

if gnn_layer_type in [GCNLayer, GATLayer]: 
    graph_type = Pps2DGraph #
else: 
    graph_type = PpsHyperGraph #

# 存放预处理好的数据的目录。dataset_dir.txt 的内容示例：E:/DataScienceDataset/
if os.path.exists('./dataset_dir.txt'):
    data_dir = os.path.join(IOHelper.GetFirstLineContent('./dataset_dir.txt'), dataset_name)
else:
    data_dir = './Data/' + dataset_name

# 存放训练结果的目录，例如：./Results/AlibabaAir-Complete-RawGnn/
result_dir = dataset_name.strip('/').split('/')
result_dir.append(model_type.__name__)
if model_type == RawGnn:
    result_dir.append(f'{gnn_layer_count}{gnn_layer_type.__name__}')
    if gnn_layer_type == IHGNNLayer:
        result_dir.append(f'O{feature_interaction_order}')
result_dir.append(f'emb{Gs.embedding_size}')
result_dir = '-'.join(result_dir)
result_dir = os.path.join('Results', result_dir)

if not os.path.exists(result_dir): 
    os.makedirs(result_dir)

time_for_fn = time.strftime('%y%m%d-%H%M%S', time.localtime())
# 存放训练日志的文件
fn_train_log = (f'{time_for_fn}_train_log.txt' if args.storemetrics else 'train_log.txt')
# 存放模型测试指标的文件
fn_metrics = f'{time_for_fn}_metrics.txt'

IOHelper.StartLogging(os.path.join(result_dir, fn_train_log))
IOHelper.LogPrint()

if True:
    IOHelper.LogPrint(f'设备：{device}')
    IOHelper.LogPrint(f'Batch 大小：{Gs.batch_size}')
    IOHelper.LogPrint(f'学习率：{Gs.learning_rate}')
    IOHelper.LogPrint(f'Embedding 大小：{Gs.embedding_size}')
    IOHelper.LogPrint(f'L2 系数：{Gs.weight_decay}')
    IOHelper.LogPrint(f'随机 / 非随机负采样数：{Gs.random_negative_sample_size} / {Gs.non_random_negative_sample_size}')

    IOHelper.LogPrint(f'\n模型：  {model_type.__name__}')
    IOHelper.LogPrint(f'数据集：{dataset_name}')
    IOHelper.LogPrint(f'预测层：{prediction_layer_type.__name__}')

    if model_type in [RawGnn]:
        IOHelper.LogPrint(f'\nGNN 网络：{gnn_layer_count} 层 {gnn_layer_type.__name__}')
        IOHelper.LogPrint(f'图类型：  {graph_type.__name__}')

        if gnn_layer_type == GCNLayer:
            IOHelper.LogPrint(f'二元图完整性：{Gs.graph_completeness}')
        if gnn_layer_type == GATLayer:
            IOHelper.LogPrint(f'GAT 权重计算激活函数：{Gs.Gnn.gat_activation[0].__name__}')
        if gnn_layer_type == IHGNNLayer:
            IOHelper.LogPrint(f'特征交互：{feature_interaction_order} 阶')
            IOHelper.LogPrint(f'二阶段注意力：{phase2_attention}')
            if phase2_attention:
                IOHelper.LogPrint(f'GAT 权重计算激活函数：{Gs.Gnn.gat_activation[0].__name__}')

    IOHelper.LogPrint(f'query 处理方式：{Gs.Query.transform}')
    if Gs.Query.transform == Gsv.activation:
        IOHelper.LogPrint(f'query 激活函数：{Gs.Query.transform_activation.__name__}')

    IOHelper.LogPrint(f'\n使用验证集：{Gs.use_valid_dataset}')

IOHelper.LogPrint()
IOHelper.LogPrint(f'存储指标：{args.storemetrics}')
IOHelper.LogPrint(f'存档模型：{args.storecheckpoint}')
IOHelper.LogPrint(f'读取存档：{args.checkpoint or False}\n')

# 构造数据集、神经网络、损失函数、优化器

IOHelper.LogPrint('正在读取数据集...')
dataset_train = GraphDataset(
    fn_graph_info=os.path.join(data_dir, 'graph_info.txt'),
    fn_queries_multihot=os.path.join(data_dir, 'queries_multihot.txt'),
    fn_train_data=os.path.join(data_dir, 'train_data.csv'),
    graph_type=graph_type,
    random_negative_sample_size=Gs.random_negative_sample_size,
    non_random_negative_sample_size=Gs.non_random_negative_sample_size,
    device=device
)

dataloader_train = DataLoader(dataset_train, Gs.batch_size, shuffle=True, collate_fn=GraphDataset.collate_fn)
dataloader_valid = TestSearchLogDataLoader(fn_search_log=os.path.join(data_dir, 'valid_data.csv'), dataset_train=dataset_train, device=device)
dataloader_test  = TestSearchLogDataLoader(fn_search_log=os.path.join(data_dir, 'test_data.csv'), dataset_train=dataset_train, device=device)

IOHelper.LogPrint('\n正在构建模型...')
if model_type == RawGnn:
    model = RawGnn(
        device=device, 
        dataset=dataset_train,
        embedding_size=Gs.embedding_size,
        gnn_layer_type=gnn_layer_type,
        gnn_layer_count=gnn_layer_count,
        predictions=prediction_layer_type,
        lambda_muq=Gs.lambda_muq_for_hem,
        feature_interaction_order=feature_interaction_order,
        phase2_attention=phase2_attention
    ).to(device)
elif model_type == Srrl:
    meta_paths = MetaPaths(dataset_train)
    train_iterator_KG = OneShotIterator(
        DataLoader(SrrlDatasetKG(meta_paths, Gs.negative_sample_size, 'tail-company-batch'), Gs.batch_size, True, collate_fn=SrrlDatasetKG.collate_fn), 
        DataLoader(SrrlDatasetKG(meta_paths, Gs.negative_sample_size, 'head-company-batch'), Gs.batch_size, True, collate_fn=SrrlDatasetKG.collate_fn),
        DataLoader(SrrlDatasetKG(meta_paths, Gs.negative_sample_size, 'query-company-batch'), Gs.batch_size, True, collate_fn=SrrlDatasetKG.collate_fn)
    )
    model = Srrl(
        dataset=dataset_train,
        embedding_size=Gs.embedding_size,
        prediction_layer_type=None,
        lambda_muq=Gs.lambda_muq_for_hem
    ).to(device)

    srrl_steps = len(meta_paths.positive_interactions) // Gs.batch_size
    if srrl_steps * Gs.batch_size < len(meta_paths.positive_interactions): srrl_steps += 1

    model.srrl_steps = srrl_steps
    model.train_iterator_KG = train_iterator_KG
else:
    raise NotImplementedError(f'不受支持的 model_type：{model_type}')

loss_function = nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), Gs.learning_rate, weight_decay=Gs.weight_decay)

# 加载训练好的模型
if args.checkpoint != '':
    if args.checkpoint == 'latest':
        files = [fn for fn in os.listdir(result_dir) if fn.startswith('checkpoint_') and os.path.isfile(os.path.join(result_dir, fn))]
        if len(files) == 0:
            raise FileNotFoundError('未找到模型存档')
        else:
            file = sorted(files)[-1]
            checkpoint = torch.load(os.path.join(result_dir, file))
            IOHelper.LogPrint(f'成功读取模型存档：{file}')
    else:
        checkpoint = torch.load(os.path.join(result_dir, args.checkpoint))
        IOHelper.LogPrint(f'成功读取模型存档：{args.checkpoint}')

    checkpoint_epoch_count = int(checkpoint['epoch_count'])
    epoch_start = checkpoint_epoch_count + 1
    IOHelper.LogPrint(f'该存档已进行 {checkpoint_epoch_count} 轮训练')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
else:
    epoch_start = 1

if not args.storecheckpoint:
    epoch_start_store_position = None
    epoch_store_frequency = None

IOHelper.LogPrint(f'\nModel Parameters ({len(list(model.parameters()))}):')
print_network_parameters(model)

# 开始训练
pc = ProcessController(
    epoch_count, 
    epoch_start, 
    epoch_start_test_position, 
    epoch_test_frequency,
    epoch_start_store_position,
    epoch_store_frequency
)
IOHelper.LogPrint(f'\n开始训练...')
IOHelper.LogPrint(f'轮数 {pc.EpochCount} | 开始测试 {epoch_start_test_position} | 测试频率 {epoch_test_frequency}' +
    f' | 开始存档 {epoch_start_store_position} | 存档频率 {epoch_store_frequency}\n')

all_metrics = MetricsCollection(Gs.use_valid_dataset)

for epoch in pc:

    avg_loss, time_train = train_and_get_avg_loss(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        dataset_train=dataset_train,
        dataloader_train=dataloader_train,
        pc=pc,
        device=device
    )

    pc.AddTrainTime(time_train)

    if pc.ShouldStore():
        fn = time.strftime(f'checkpoint_%y%m%d-%H%M%S_epoch{pc.CurrentEpoch}', time.localtime())
        IOHelper.LogPrint('\n阶段训练完成，正在存档至：' + os.path.join(result_dir, fn))
        torch.save(
            { 
                'epoch_count': pc.CurrentEpoch, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict() 
            },
            os.path.join(result_dir, fn)
        )

    if pc.ShouldTest():

        IOHelper.LogPrint('\n开始在 测试集 上测试...')
        u_metrics, m_t, time_t = test_and_get_avg_metrics(model, dataset_train, dataloader_test, bool(Gs.long_tail_stat_fn))

        if Gs.long_tail_stat_fn:
            with open(os.path.join(result_dir, Gs.long_tail_stat_fn), 'w', encoding='utf-8') as f:
                for u, mu in enumerate(u_metrics):
                    if mu is None:
                        line = f'{u},{len(dataset_train.user_history_items[u])},,,'
                    else:
                        line = ','.join([str(u), str(len(dataset_train.user_history_items[u]))] + mu.to_string(no_title=True).split(' '))
                    f.write(line)
                    f.write('\n')

        if Gs.use_valid_dataset:
            IOHelper.LogPrint('开始在 验证集 上测试...')
            _, m_v, time_v = test_and_get_avg_metrics(model, dataset_train, dataloader_valid)
            all_metrics.add(pc.CurrentEpoch, m_t, m_v)
            pc.AddTestTime(time_t + time_v)
        else:
            all_metrics.add(pc.CurrentEpoch, m_t)
            pc.AddTestTime(time_t)
            
        if args.storemetrics:
            with open(os.path.join(result_dir, fn_metrics), 'a', encoding='utf-8') as fout:
                fout.write(f'Epoch {pc.CurrentEpoch} Avg loss {avg_loss:.4f}\n{m_t.to_string()}\n')

if Gs.use_valid_dataset:
    best_epoch, best_test, best_valid = all_metrics.get_valid_best(key=lambda m: m.NDCG_at10)
    IOHelper.LogPrint(f'Best valid metrics at epoch \033[0;44m{best_epoch}\033[0m:')
    IOHelper.LogPrint(best_valid.to_string(highlight=True), put_time_in_single_line=True)
    IOHelper.LogPrint('Corresponding test metrics:')
    IOHelper.LogPrint(best_test.to_string(highlight=True), put_time_in_single_line=True)
else:
    best_epoch, best_test = all_metrics.get_test_best(key=lambda m: m.NDCG_at10)
    IOHelper.LogPrint(f'Best test metrics at epoch \033[0;44m{best_epoch}\033[0m:')
    IOHelper.LogPrint(best_test.to_string(highlight=True), put_time_in_single_line=True)
IOHelper.LogPrint()

if args.storemetrics:
    with open(os.path.join(result_dir, fn_metrics), 'a', encoding='utf-8') as fout:
        
        if Gs.use_valid_dataset:
            fout.write(f'\n\nBest valid metrics at epoch {best_epoch}:\n')
            fout.write(best_valid.to_string())
            fout.write('\nCorresponding test metrics:\n')
        else:
            fout.write('\nBest test metrics:\n')
        fout.write(best_test.to_string())
        fout.write('\n')

        fout.write('\n\nAll TEST metrics:\n')
        fout.write(f'Epoch {Metrics.title}\n')
        for e, m_t in all_metrics.iter_epoch_test():
            fout.write(str(e) + ' ' + m_t.to_string(no_title=True) + '\n')

        if Gs.use_valid_dataset:
            fout.write('\n\nAll VALID metrics:\n')
            fout.write(f'Epoch {Metrics.title}\n')
            for e, _, m_v in all_metrics.iter_epoch_test_valid():
                fout.write(str(e) + ' ' + m_v.to_string(no_title=True) + '\n')

IOHelper.EndLogging()

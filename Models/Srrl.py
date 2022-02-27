from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable

from Dataset import GraphDataset
from Models.CommonLayers import Aggregation, MLP
from Models.PredictionLayers import HemPredictionLayer
from Models.EmbeddingLayers import EmbeddingLayer
from Helpers.GlobalSettings import Gs, Gsv
from Helpers.Torches import *
from SrrlDataset import OneShotIterator

class Srrl(nn.Module):

    _saved_u_kg: Tensor = None
    _saved_q_kg: Tensor = None
    _saved_i_kg: Tensor = None
    _saved_u_ps: Tensor = None
    _saved_i_ps: Tensor = None

    srrl_steps: int
    train_iterator_KG: OneShotIterator
    
    def __init__(self, 
        dataset: GraphDataset,
        embedding_size: int, 
        prediction_layer_type: Type,
        lambda_muq: float):
        super().__init__()

        self.dataset = dataset
        self.embedding_size = embedding_size
        self.prediction_layer_type = prediction_layer_type

        self.KG = EmbeddingLayer(
            dataset=dataset,
            embedding_size=embedding_size
        )
        # 不要使用 PS_embed 里面的 query embedding 模块
        self.PS = EmbeddingLayer(
            dataset=dataset,
            embedding_size=embedding_size
        )
        self.PS.embedding_bag_vocabulary = None

        self.init_embedding(self.KG.embedding_user)
        self.init_embedding(self.KG.embedding_item)
        self.init_embedding(self.KG.embedding_bag_vocabulary)
        self.init_embedding(self.PS.embedding_user)
        self.init_embedding(self.PS.embedding_item)

        self.kg_aggre_head = Aggregation(embedding_size * 2, embedding_size)
        self.kg_aggre_query = Aggregation(embedding_size * 2, embedding_size)
        self.kg_aggre_tail = Aggregation(embedding_size * 2, embedding_size)
        self.kg_mlp_pre = MLP(embedding_size * 2, embedding_size)

        self.g_u = Aggregation(embedding_size * 2, embedding_size)
        self.g_i = Aggregation(embedding_size * 2, embedding_size)

        if prediction_layer_type == HemPredictionLayer:
            self.prediction_layer = HemPredictionLayer(
                feature_dimension=embedding_size,
                lambda_muq=lambda_muq,
                item_count=dataset.item_count
            )
        else:
            self.ps_mlp_uq = MLP(embedding_size * 2, embedding_size)
            self.ps_mlp_ui = MLP(embedding_size * 2, embedding_size)
            self.ps_mlp_pred = MLP(embedding_size*2, 1)
        
        self.init_parameters()


    def trainkg(self, sample, mode: str, positive_mode: bool):

        q_emb = self.KG.embed_query(sample[0][:, 1]).unsqueeze(1)

        if positive_mode:
            positive_sample, items2_positive, users2_positive, queries2_positive = sample

            u_emb      = self.KG.embed_user(positive_sample[:, 0]).unsqueeze(1)
            u2_pos_emb = self.KG.embed_user(users2_positive[:, 0]).unsqueeze(1)
            q2_pos_emb = self.KG.embed_query(queries2_positive[:, 0]).unsqueeze(1)
            i_emb      = self.KG.embed_item(positive_sample[:, 2]).unsqueeze(1)
            i2_pos_emb = self.KG.embed_item(items2_positive[:, 0]).unsqueeze(1)

            score = self.compat_fun(u_emb, q_emb, i_emb, None, i2_pos_emb, u2_pos_emb, q2_pos_emb, mode, positive_mode)
        else:
            positive_sample, i2_neg, items2_positive, users2_positive, queries2_positive = sample

            u_emb      = self.KG.embed_user(positive_sample[:, 0]).unsqueeze(1)
            u2_pos_emb = self.KG.embed_user(users2_positive[:, 0]).unsqueeze(1)
            q2_pos_emb = self.KG.embed_query(queries2_positive[:, 0]).unsqueeze(1)
            i_emb      = self.KG.embed_item(positive_sample[:, 2]).unsqueeze(1)
            i2_neg_emb = self.KG.embed_item(i2_neg.long())
            i2_pos_emb = self.KG.embed_item(items2_positive[:, 0]).unsqueeze(1)

            score = self.compat_fun(u_emb, q_emb, i_emb, i2_neg_emb, i2_pos_emb, u2_pos_emb, q2_pos_emb, mode, positive_mode)

        return score
        

    def forward(self, uids: LongTensor, queries: LongTensor, items: LongTensor = None):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'tail-batch', 'head-company-batch' or 'tail-company-batch' mode, sample consists four part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        the third part is the tail true company.
        the fourth part is the head true company.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, query) or (query, tail)).
        '''

        if self._saved_i_kg is None:
            if Gs.Srrl.KG_loss:
                kg_user_embedded = self.KG.embed_user(uids).clone().detach()
                kg_item_embedded = self.KG.embed_item(items).clone().detach()
                ps_user_embedded = self.PS.embed_user(uids)
                ps_item_embedded = self.PS.embed_item(items)

                q_latent = self.KG.embed_query(queries)

                u_latent = torch.cat([ps_user_embedded, kg_user_embedded], -1)
                i_latent = torch.cat([ps_item_embedded, kg_item_embedded], -1)
                u_latent = F.normalize(u_latent, dim=-1)
                i_latent = F.normalize(i_latent, dim=-1)
                u_latent = self.g_u(u_latent)
                i_latent = self.g_i(i_latent)
            else:
                u_latent = self.PS.embed_user(uids)
                q_latent = self.KG.embed_query(queries)
                i_latent = self.PS.embed_item(items)
        else:
            if Gs.Srrl.KG_loss:
                kg_user_embedded = self._saved_u_kg[uids]
                kg_item_embedded = (self._saved_i_kg if items is None else self._saved_i_kg[items])
                ps_user_embedded = self._saved_u_ps[uids]
                ps_item_embedded = (self._saved_i_ps if items is None else self._saved_i_ps[items])

                q_latent = self._saved_q_kg[queries]

                u_latent = torch.cat([ps_user_embedded, kg_user_embedded], -1)
                i_latent = torch.cat([ps_item_embedded, kg_item_embedded], -1)
                u_latent = F.normalize(u_latent, dim=-1)
                i_latent = F.normalize(i_latent, dim=-1)
                u_latent = self.g_u(u_latent)
                i_latent = self.g_i(i_latent)
            else:
                u_latent = self._saved_u_ps[uids]
                q_latent = self._saved_q_kg[queries]
                i_latent = (self._saved_i_ps if items is None else self._saved_i_ps[items])

        if self.prediction_layer_type == HemPredictionLayer:
            scores = self.prediction_layer(u_latent, q_latent, i_latent, items)
        else:
            uq_latent = self.ps_mlp_uq(F.normalize(torch.cat([u_latent, q_latent], dim=-1), dim=-1))
            ui_latent = self.ps_mlp_ui(F.normalize(torch.cat([u_latent, i_latent], dim=-1), dim=-1))
            scores = self.ps_mlp_pred(F.normalize(torch.cat([uq_latent, ui_latent], dim=-1), dim=-1)).squeeze(-1)

        return scores
    

    def save_features_for_test(self) -> None:
        '''在测试模式（无梯度）下，保存所有特征以加速测试。'''
        if Gs.Srrl.KG_loss:
            self._saved_u_kg, self._saved_q_kg, self._saved_i_kg = self.KG()
            self._saved_u_ps, self._saved_i_ps = self.PS.embed_user(), self.PS.embed_item()
        else:
            self._saved_u_ps, self._saved_i_ps = self.PS.embed_user(), self.PS.embed_item()
            self._saved_q_kg = self.KG.embed_query()
    
    def clear_saved_feature(self) -> None:
        self._saved_u_kg, self._saved_q_kg, self._saved_i_kg = None, None, None
        self._saved_u_ps, self._saved_i_ps = None, None


    def compat_fun(self, users, queries, items, items_neg, items2_pos, users2_pos, queries2_pos, mode: str, positive_mode: bool):

        if positive_mode:
            if mode == 'tail-company-batch':
                score = self.trans_u_q_i_i2(users, queries, items, items2_pos)
            elif mode == 'head-company-batch':
                score = self.trans_u_u2_q_i(users, queries, items, users2_pos)
            elif mode == 'query-company-batch':
                score = self.trans_u_q_q2_i(users, queries, items, queries2_pos)
        else:
            if mode == 'tail-company-batch':
                score = self.trans_u_q_i_i2(users, queries, items_neg, items)
            elif mode == 'head-company-batch':
                score = self.trans_u_u2_q_i(users, queries, items_neg, users2_pos)
            elif mode == 'query-company-batch':
                score = self.trans_u_q_q2_i(users, queries, items_neg, queries2_pos)

        return score

    def trans_u_q_i_i2(self, users, queries, items, items2_positive):

        if items2_positive.size()[1] == items.size()[1]: # positive_mode
            items_concat = torch.cat((items, items2_positive), -1)
        else: # negative_mode
            items2_positive = items2_positive.expand(-1, items.size()[1], -1)
            items_concat = torch.cat((items, items2_positive), -1)
        items_concat = F.normalize(items_concat, dim=-1)
        items_concat = self.kg_aggre_tail(items_concat)

        users_queries_concat = torch.cat([users, queries], -1)
        users_queries_concat = F.normalize(users_queries_concat, dim=-1)
        users_queries_concat = self.kg_mlp_pre(users_queries_concat)

        score = torch.sum(items_concat * users_queries_concat, dim=2)
        return score

    def trans_u_u2_q_i(self, users, queries, items, users2_positive):

        users_concat = torch.cat((users, users2_positive), -1)
        users_concat = F.normalize(users_concat, dim=-1)

        users_queries_concat = torch.cat([self.kg_aggre_head(users_concat), queries], -1)
        users_queries_concat = F.normalize(users_queries_concat, dim=-1)
        users_queries_concat = self.kg_mlp_pre(users_queries_concat)

        score = torch.sum(items * users_queries_concat, dim=2)
        return score

    def trans_u_q_q2_i(self, users, queries, items, queries2_positive):

        queries_concat = torch.cat((queries, queries2_positive), -1)
        queries_concat = F.normalize(queries_concat, dim=-1)

        users_queries_concat = torch.cat([users, self.kg_aggre_query(queries_concat)], -1)
        users_queries_concat = F.normalize(users_queries_concat, dim=-1)
        users_queries_concat = self.kg_mlp_pre(users_queries_concat)
        
        score = torch.sum(items * users_queries_concat, dim=2)
        return score

    def init_embedding(self, emb: nn.Embedding):
        emb.weight.data = F.normalize(emb.weight.data, p=2, dim=1)
    
    def init_parameters(self, method='xavier', exclude='embedding', seed=123):
        for name, w in self.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass
        
from typing import List, Dict, Optional, Set, Any, Tuple, Type

from Dataset import GraphDataset
from Models.EmbeddingLayers import EmbeddingLayer
from Models.GnnLayers import GCNLayer, GATLayer, HGCNLayer, IHGNNLayer
from Models.PredictionLayers import HemPredictionLayer
from Helpers.Torches import *
from Helpers.GlobalSettings import Gs, Gsv

class RawGnn(nn.Module):

    _saved_output_feature: Tensor = None
    
    def __init__(self, 
        device: torch.device, 
        dataset: GraphDataset,
        embedding_size: int, 
        gnn_layer_type: Type,
        gnn_layer_count: int,
        feature_interaction_order: int,
        phase2_attention: bool,
        predictions: Type,
        lambda_muq: float):

        """初始化一个模型。

        参数：
            device:                     设备。
            embedding_size:             对 user, query 或 item 做 embedding 时的目标向量维度。
            gcn_layer_count:            GCN 网络层数。
            users_onehot:               一维张量。
            queries_multihot:           二维稀疏张量，每行表示一个 query，列表示 one-hot 维度。
            items_onehot:               一维张量。
            adjacency:                  二维稀疏张量，表示邻接矩阵。对角线上的元素须为零。
            vocabulary_onehot:          query 所用的词典，是一维张量。
        """
        super().__init__()

        # 记录参数
        self.device                     = device
        self.dataset                    = dataset
        self.embedding_size             = embedding_size
        self.gnn_layer_type             = gnn_layer_type
        self.gnn_layer_count            = gnn_layer_count
        self.feature_interaction_order  = feature_interaction_order
        self.phase2_attention           = phase2_attention
        self.prediction_layer_type      = predictions

        self.output_feature_size        = embedding_size * (1 + self.gnn_layer_count)

        # Embedding 层
        self.embeddings = EmbeddingLayer(
            dataset=dataset,
            embedding_size=embedding_size
        )

        # GNN 网络层，多层
        self.gnns = []
        for layer in range(self.gnn_layer_count):
            if gnn_layer_type in [
                GCNLayer,
                GATLayer,
                HGCNLayer
            ]:
                self.gnns.append(
                    gnn_layer_type(
                        device=device,
                        dataset=dataset,
                        input_dimension=embedding_size,
                        output_dimension=embedding_size
                    )
                )
            elif gnn_layer_type in [
                IHGNNLayer
            ]:
                fi_order_here = feature_interaction_order
                if fi_order_here > 1 and layer > 0:
                    fi_order_here = 1
                self.gnns.append(
                    gnn_layer_type(
                        device=device,
                        dataset=dataset,
                        input_dimension=embedding_size,
                        output_dimension=embedding_size,
                        feature_interaction_order=fi_order_here,
                        phase2_attention=phase2_attention
                    )
                )
            else:
                raise NotImplementedError(f'不支持的 GNN 网络层类型：{gnn_layer_type}')
        for i, gnn in enumerate(self.gnns): self.add_module(f'gnn_{i}', gnn)

        # 预测层
        if predictions == HemPredictionLayer:
            self.prediction_layer = HemPredictionLayer(
                feature_dimension=self.output_feature_size,
                lambda_muq=lambda_muq,
                item_count=dataset.item_count
            )
        else:
            raise NotImplementedError(f'不支持的预测层类型：{predictions}')


    def forward(self, user_indices: Tensor, query_indices: Tensor, item_indices: Optional[Tensor] = None):

        """参数：这里的 u q i 并非其在邻接矩阵的结点列表中的索引，而是从 0 开始的。\n
        返回值：由每个 interaction 发生的可能性分数构成的一维张量。"""

        # 训练模式下
        if self._saved_output_feature is None:
            # 计算图中所有结点的特征，生成结点特征矩阵 X
            input_features = torch.cat(self.embeddings(None, None, None))
            # 算 GCN 输出，把所有输出拼接起来，得到输出的结点特征矩阵 X'
            gnn_outputs = [input_features]
            gnn_output = input_features
            for gnn in self.gnns:
                gnn_output = gnn(gnn_output)
                gnn_outputs.append(gnn_output)
            # 需要测量高阶特征权重时，直接中断操作
            if Gs.Debug._calculate_highorder_info:
                return
            output_feature = torch.cat(gnn_outputs, 1)
        # 测试模式下
        else:
            output_feature = self._saved_output_feature

        # 分别提取 user query item 的 feature
        output_user_feature = output_feature[user_indices]
        output_query_feature = output_feature[query_indices + self.dataset.query_start_index_in_graph]
        if item_indices is not None: 
            output_item_feature = output_feature[item_indices + self.dataset.item_start_index_in_graph]
        else: 
            output_item_feature = output_feature[self.dataset.item_start_index_in_graph:]
        
        # 做预测
        if self.prediction_layer_type == HemPredictionLayer:
            similarity: Tensor = self.prediction_layer(
                output_user_feature,
                output_query_feature,
                output_item_feature,
                item_indices
            )

        return similarity
    

    def save_features_for_test(self) -> None:
        '''在测试模式（无梯度）下，保存所有 GNN 网络层的输出以加速测试。'''
        input_features = torch.cat(self.embeddings(None, None, None))
        gnn_outputs = [input_features]
        gnn_output = input_features
        for gnn in self.gnns:
            gnn_output = gnn(gnn_output)
            gnn_outputs.append(gnn_output)
        self._saved_output_feature = torch.cat(gnn_outputs, 1)
    
    def clear_saved_feature(self) -> None:
        self._saved_output_feature = None


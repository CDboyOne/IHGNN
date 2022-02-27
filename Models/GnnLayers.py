from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable

from Helpers.Graph import Pps2DGraph, PpsHyperGraph, PpsLogHyperGraph
from Helpers.Torches import *
from Helpers.GlobalSettings import Gs, Gsv
from Models.CommonLayers import FeatureInteractor
from Dataset import GraphDataset

class GCNLayer(nn.Module):

    def __init__(self, 
        device:           th.device,
        dataset:          GraphDataset,
        input_dimension:  int,
        output_dimension: int):

        super().__init__()
        self.device = device
        self.dataset = dataset
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.adjacency = SparseTensor.from_torch_sparse_coo_tensor(dataset.graph2d.Adjacency).coalesce()
        self.Dv_neg_1_slash_2 = dataset.graph2d.VertexDegrees.pow(-0.5)
        self.feature_transform = nn.Linear(input_dimension, output_dimension)


    def forward(self, input_features: Tensor):
        
        # DADX(W)
        all: Tensor = input_features

        # 为了减小运算量，当输出维度更小时，就先进行特征转换
        if self.input_dimension >= self.output_dimension:
            all = self.feature_transform(all)
            all = self.Dv_neg_1_slash_2 * all
            all = thsp.matmul(self.adjacency, all)
            all = self.Dv_neg_1_slash_2 * all
        else:
            all = self.Dv_neg_1_slash_2 * all
            all = thsp.matmul(self.adjacency, all)
            all = self.Dv_neg_1_slash_2 * all
            all = self.feature_transform(all)
        
        return all


class GATLayer(nn.Module):

    def __init__(self,
        device:           th.device,
        dataset:          GraphDataset,
        input_dimension:  int,
        output_dimension: int):

        super().__init__()
        self.device = device
        self.dataset = dataset
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        indices = dataset.graph2d.Adjacency.indices()

        self.node_feature_selector = indices.t()
        if Gs.Gnn.gat_head == Gsv.concat:
            feature_aggregate = nn.Linear(2 * output_dimension, 1)
            init.xavier_uniform_(
                feature_aggregate.weight, 
                gain=init.calculate_gain(Gs.Gnn.gat_activation[1])
            )
            self.feature_aggregate = nn.Sequential(
                feature_aggregate,
                Gs.Gnn.gat_activation[0]()
            )
        elif Gs.Gnn.gat_head == Gsv.product:
            feature_aggregate = nn.Linear(output_dimension, 1)
            init.xavier_uniform_(
                feature_aggregate.weight,
                gain=init.calculate_gain(Gs.Gnn.gat_activation[1])
            )
            self.feature_aggregate = nn.Sequential(
                feature_aggregate,
                Gs.Gnn.gat_activation[0](),
            )
        else:
            raise ValueError()

        self.dgl_graph = dgl.graph(
            data=(indices[0], indices[1]), 
            num_nodes=dataset.node_count,
            device=device)
        
        self.feature_transform = nn.Linear(input_dimension, output_dimension)
    

    def forward(self, input_features: Tensor):

        input_features = self.feature_transform(input_features)
        # (edge_count, 2, feature_dimension)
        selected_node_feature = input_features[self.node_feature_selector]
        
        if Gs.Gnn.gat_head == Gsv.concat:
            # (edge_count, 2 * feature_dimension)
            selected_node_feature = selected_node_feature.reshape(-1, 2 * self.output_dimension)
        elif Gs.Gnn.gat_head == Gsv.product:
            # (edge_count, feature_dimension)
            selected_node_feature = selected_node_feature[:, 0, :] * selected_node_feature[:, 1, :]
            # selected_node_feature = F.normalize(selected_node_feature)

        # (edge_count)
        edge_importance = self.feature_aggregate(selected_node_feature).squeeze_()
        edge_importance = dgl.ops.edge_softmax(self.dgl_graph, edge_importance.float())

        output_features = dgl.ops.u_mul_e_sum(self.dgl_graph, input_features.float(), edge_importance)
        return output_features


class HGCNLayer(nn.Module):

    def __init__(self,
        device: torch.device,
        dataset: GraphDataset,
        input_dimension: int,
        output_dimension: int):

        super().__init__()
        self.device = device
        self.dataset = dataset

        graph = dataset.graph
        incidence = graph.Adjacency

        self.Dv_neg_1_slash_2 = graph.VertexDegrees.pow(-0.5)
        self.De_neg_1 = graph.EdgeDegrees.pow(-1)

        self.incidence   = SparseTensor.from_torch_sparse_coo_tensor(incidence).coalesce()
        self.incidence_t = SparseTensor.from_torch_sparse_coo_tensor(incidence.t()).coalesce()

        self.feature_transform = nn.Linear(input_dimension, output_dimension)
    

    def forward(self, input_features: Tensor):

        # HyperGCN: Dv W(h) H De Ht W(h)(t) Dv X W(x)
        input_features = self.feature_transform(input_features)
        input_features = self.Dv_neg_1_slash_2 * input_features

        edge_features = thsp.matmul(self.incidence_t, input_features)
        edge_features = self.De_neg_1 * edge_features
        
        output_features = thsp.matmul(self.incidence, edge_features)
        output_features = self.Dv_neg_1_slash_2 * output_features
        return output_features


class IHGNNLayer(nn.Module):

    class _FakeGraph:
        def __init__(self, adjacency: Tensor) -> None:
            self.Adjacency = adjacency
    class _FakeDataset:
        def __init__(self, adjacency: Tensor, node_count: int) -> None:
            self.Graph = IHGNNLayer._FakeGraph(adjacency)
            self.NodeCount = node_count

    def __init__(self,
        device:                     torch.device,
        dataset:                    GraphDataset,
        input_dimension:            int,
        output_dimension:           int,
        feature_interaction_order:  int,
        phase2_attention:           bool):

        super().__init__()
        self.device = device
        self.dataset = dataset
        self.feature_interaction_order = feature_interaction_order
        self.attention_phase2 = phase2_attention

        graph = dataset.hypergraph
        assert feature_interaction_order in [1, 2, 3], '特征交互阶数只能为 1 2 或 3'
        incidence = graph.Adjacency
        node_indices = incidence.indices()[0]
        edge_indices = incidence.indices()[1]

        #self.Dv_neg_1_slash_2 = graph.VertexDegrees.pow(-0.5)
        self.Dv_neg_1 = graph.VertexDegrees.pow(-1)
        #self.De_neg_1 = graph.EdgeDegrees.pow(-1)

        self.incidence = SparseTensor.from_torch_sparse_coo_tensor(incidence).coalesce()

        # 边特征聚合层 或 高阶特征层
        self.feature_interactor = FeatureInteractor(
            dataset=dataset,
            max_order=feature_interaction_order,
            node_feature_dimension=input_dimension,
            output_dimension=input_dimension
        )
        
        if phase2_attention:
            # 构造第二阶段的二元图
            fake_adj = torch.sparse_coo_tensor(
                indices=torch.stack([edge_indices + dataset.node_count, node_indices]),
                values=torch.ones(len(edge_indices), dtype=torch.long, device=device),
                size=[dataset.node_count + graph.EdgeCount] * 2,
                dtype=torch.float,
            ).coalesce()
            self.fake_gat = GATLayer(
                device=device,
                dataset=IHGNNLayer._FakeDataset(
                    fake_adj,
                    dataset.node_count + graph.EdgeCount
                ),
                input_dimension=output_dimension,
                output_dimension=output_dimension
            )
        
        self.feature_transform = nn.Linear(input_dimension, output_dimension)
    

    def forward(self, input_features: Tensor):

        # HyperGCN: Dv W(h) H De Ht W(h)(t) Dv X W(x)
        input_features = self.feature_transform(input_features)
        edge_features = self.feature_interactor(input_features)

        if self.attention_phase2:
            # Phase-2 attention
            output_features = self.fake_gat(torch.cat([input_features, edge_features]))
            output_features = output_features[:input_features.size(0)]
        else:
            # High-order feature interaction
            output_features = thsp.matmul(self.incidence, edge_features)
            output_features = self.Dv_neg_1 * output_features
        
        return output_features

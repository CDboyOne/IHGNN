from typing import Iterator, List, Dict, Set, Any, Tuple

from Dataset import GraphDataset
from Helpers.GlobalSettings import Gs, Gsv
from Helpers.Torches import *

class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, output_dim)
        )
    def forward(self, x): return self.mlp(x)

class Aggregation(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.aggregation = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU()
        )
    def forward(self, x): return self.aggregation(x)


class FeatureInteractor(nn.Module):

    def __init__(self,
        dataset: GraphDataset,
        max_order: int,
        node_feature_dimension: int,
        output_dimension: int):

        super().__init__()
        self.max_order              = max_order
        self.node_feature_dimension = node_feature_dimension
        self.output_dimension       = output_dimension

        if max_order == 1:
            self.aggregation = nn.Linear(3 * node_feature_dimension, output_dimension)
            self.edge_feature_selector = dataset.graph.I3
        elif max_order in [2, 3]:
            self.edge_user_indices  = dataset.graph.I3[:, 0]
            self.edge_query_indices = dataset.graph.I3[:, 1]
            self.edge_item_indices  = dataset.graph.I3[:, 2]

            if max_order == 2:
                inter_dimension = 6 * node_feature_dimension
            else:
                inter_dimension = 7 * node_feature_dimension
            
            self.aggregation = nn.Linear(inter_dimension, output_dimension)
    

    def forward(self, node_features: Tensor):

        if self.max_order == 1:
            # [E, 3, feature_dim]
            selected_node_feature = node_features[self.edge_feature_selector]
            # [E, 3 * feature_dim]
            edge_features = selected_node_feature.reshape(-1, 3 * node_features.shape[1])
            # [E, feature_dim]
            edge_features: Tensor = self.aggregation(edge_features)
            
        else:
            # [E, feature_dim]
            u = node_features[self.edge_user_indices]
            q = node_features[self.edge_query_indices]
            i = node_features[self.edge_item_indices]

            uq = u * q
            qi = q * i
            iu = i * u

            if self.max_order == 3:
                uqi = uq * i

            if self.max_order == 2:
                edge_features = th.cat([u, q, i, uq, qi, iu], 1)
            else:
                edge_features = th.cat([u, q, i, uq, qi, iu, uqi], 1)
            edge_features = self.aggregation(edge_features)
        
        return edge_features


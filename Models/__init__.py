from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
from Models.RawGnn import RawGnn
from Models.Srrl import Srrl
from Models.EmbeddingLayers import EmbeddingLayer
from Models.GnnLayers import GCNLayer, GATLayer, HGCNLayer, IHGNNLayer
from Models.PredictionLayers import HemPredictionLayer
from Models.CommonLayers import FeatureInteractor, Aggregation, MLP

PpsModel = Union[RawGnn, Srrl]
PpsModelTypes = [RawGnn, Srrl]

GnnLayer = Union[GCNLayer, GATLayer, HGCNLayer, IHGNNLayer]
GnnLayerTypes = [GCNLayer, GATLayer, HGCNLayer, IHGNNLayer]

parse_model_type: Dict[str, PpsModel] = { t.__name__ : t for t in PpsModelTypes }
parse_model_type.update({ t.__name__.lower() : t for t in PpsModelTypes })
parse_model_type.update({ t.__name__.upper() : t for t in PpsModelTypes })
parse_model_type[''] = None

parse_gnn_layer: Dict[str, GnnLayer] = { t.__name__ : t for t in GnnLayerTypes }
parse_gnn_layer.update({ t.__name__ : t for t in GnnLayerTypes })
parse_gnn_layer.update({ t.__name__.strip('Layer') : t for t in GnnLayerTypes })
parse_gnn_layer.update({ t.__name__.strip('Layer').lower() : t for t in GnnLayerTypes })
parse_gnn_layer[''] = None
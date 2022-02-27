from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable, Callable
from Helpers.Torches import *

class Gsv:

    mean = 'mean'
    activation = 'activation'
    rnn = 'rnn'

    concat = 'concatenation'
    product = 'product'

    graph_uqi = 'uqi'
    graph_only_uq = 'uq'
    graph_only_ui = 'ui'
    graph_only_qi = 'qi'

class Gs:

    use_valid_dataset: bool = True
    adjust_learning_rate: bool = True # 是否在 loss 非常小时降低学习率

    lambda_muq_for_hem = 0.5

    # 指一个 batch 中正样本的数量，例如为 100 且负采样倍数为 10，则一个 batch 有 1100 行
    batch_size = 100
    # DO NOT modify this
    batch_size_times = 1
    learning_rate = 0.001
    embedding_size = 32
    weight_decay = 0#1e-5

    # 目前仅支持 2D-Graph
    graph_completeness: str

    # 各个用户的平均评价指标的统计信息，为 None 表示不统计
    long_tail_stat_fn: str = None

    random_negative_sample_size     = 10
    non_random_negative_sample_size = 0
    negative_sample_size = random_negative_sample_size + non_random_negative_sample_size

    # 特征交互，仅用于 HyperGCN HyperGAT
    class HighOrderFeature:

        alpha2 = [0, 1]
        alpha2 = [0.01, 0.99]
        alpha2 = [0.001, 0.999]
        alpha2 = [0.0001, 0.9999]
        alpha2 = [0.00001, 0.99999]
        alpha2 = [0.5, 0.5]
        alpha2 = [0.99, 0.01]
        alpha2 = [0.6, 0.4]
        alpha2 = [0.8, 0.2]
        alpha2 = [1, 1]

        alpha3 = [1 / 3] * 3
    
    class Gnn:

        gat_head = Gsv.product
        gat_head = Gsv.concat
        
        gat_activation = (nn.Tanh, 'tanh')
        gat_activation = (nn.ReLU, 'relu')
        gat_activation = (nn.LeakyReLU, 'leaky_relu')

    class Query:

        # 如何对输入的 query 特征进行转换
        transform = Gsv.rnn #
        transform = Gsv.activation #
        transform = Gsv.mean #
        
        transform_activation = nn.Tanh
        transform_activation = nn.ReLU
    
    class Prediction:

        # 使用余弦相似度，否则使用点积。结论：点积更好
        use_cosine_similarity: bool = False
    
    class Tem:

        encoder_count: int = 1
    
    class Srrl:

        KG_loss: bool = True
        uni_weight: bool = False
        regularization: float = 0#.00001
    
    class Dataset:

        user_history_limit: int = -1
        user_history_limit: int = 500
    
    class Debug:

        # 是否输出高阶特征的绝对值均值和标准差信息
        show_highorder_embedding_info: bool = False
        # 存储各层 GNN 网络高阶特征的绝对值均值和对应阶的权重矩阵的绝对值均值的列表，按顺序为 1 2 3...阶特征
        highorder_info: List[List[Tuple[float, float]]]
        # 存储三个 Embedding 对象的权重绝对值均值
        embedding_info: Tuple[float, float, float]
        # Do not modify this
        _calculate_embedding_info: bool = False
        # Do not modify this
        _calculate_highorder_info: bool = False
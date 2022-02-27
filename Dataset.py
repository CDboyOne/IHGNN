from typing import Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import random

from Helpers.Torches import *
from Helpers.SearchLog import SearchLog, PosInteraction
from Helpers.Graph import PpsGraph, Pps2DGraph, PpsHyperGraph, PpsLogHyperGraph
from Helpers.IOHelper import IOHelper
from Helpers.SearchLogCollection import SearchLogCollection
from Helpers.GlobalSettings import Gs, Gsv

class GraphDataset(Dataset):

    users_onehot: LongTensor
    items_onehot: LongTensor
    vocabulary_onehot: LongTensor

    queries_multihot: Tensor
    queries_for_embeddingbag: LongTensor
    queries_offset_for_embeddingbag: LongTensor

    node_count: int
    user_count: int
    query_count: int
    item_count: int
    vocab_size: int

    query_start_index_in_graph: int
    item_start_index_in_graph: int

    search_logs: SearchLogCollection
    pos_interactions: List[PosInteraction]
    neg_interactions: List[Tuple[int, int, int]]
    neg_items_for_user_query_pair: Dict[Tuple[int, int], List[int]]

    neg_sample_size: int
    rand_neg_sample_size: int
    nonrand_neg_sample_size: int

    _graph2d: Pps2DGraph = None
    _hgraph: PpsHyperGraph = None
    _hloggraph: PpsLogHyperGraph = None
    _u_his_q: List[List[int]] = None
    _u_his_i: List[List[int]] = None
    _u_his_q_tensor: LongTensor = None
    _u_his_i_tensor: LongTensor = None
    _u_his_q_maxlen: int = None
    _u_his_i_maxlen: int = None

    @property
    def user_history_queries(self) -> List[List[int]]:
        '''用户的搜索记录，按搜索时间排序。'''
        if self._u_his_q is None: self._get_user_history()
        return self._u_his_q
    @property
    def user_history_items(self) -> List[List[int]]:
        '''用户所有交互过的物品，按交互时间排序。'''
        if self._u_his_i is None: self._get_user_history()
        return self._u_his_i
    @property
    def user_history_queries_tensor(self) -> LongTensor:
        '''用户的搜索记录，按搜索时间排序。'''
        if self._u_his_q_tensor is None: self._get_user_history_tensor()
        return self._u_his_q_tensor
    @property
    def user_history_items_tensor(self) -> LongTensor:
        '''用户的搜索记录，按搜索时间排序。'''
        if self._u_his_i_tensor is None: self._get_user_history_tensor()
        return self._u_his_i_tensor
    @property
    def user_history_query_maxlen(self) -> int: 
        if self._u_his_q_maxlen is None: self._get_user_history()
        return self._u_his_q_maxlen
    @property
    def user_history_item_maxlen(self) -> int: 
        if self._u_his_i_maxlen is None: self._get_user_history()
        return self._u_his_i_maxlen

    @property
    def graph(self) -> Union[Pps2DGraph, PpsHyperGraph, PpsLogHyperGraph]:
        if self.graph_type == Pps2DGraph: return self.graph2d
        elif self.graph_type == PpsHyperGraph: return self.hypergraph
        elif self.graph_type == PpsLogHyperGraph: return self.hypergraph_log
    @property
    def graph2d(self) -> Pps2DGraph:
        if self._graph2d is None:
            self._graph2d = Pps2DGraph.from_interactions(
                self.pos_interactions, self.node_count, self.user_count, self.query_count, False, GraphDataset.device
            )
        return self._graph2d
    @property
    def hypergraph(self) -> PpsHyperGraph:
        if self._hgraph is None:
            self._hgraph = PpsHyperGraph.from_interactions(
                self.pos_interactions, self.node_count, self.user_count, self.query_count, GraphDataset.device
            )
        return self._hgraph
    @property
    def hypergraph_log(self) -> PpsLogHyperGraph:
        if self._hloggraph is None:
            self._hloggraph = PpsLogHyperGraph.from_search_logs(
                self.search_logs, self.node_count, self.user_count, self.query_count, GraphDataset.device
            )
        return self._hloggraph

    def __len__(self): return len(self.pos_interactions)
    
    def __getitem__(self, index: int) -> Tuple[Tuple[int, int, int, int], List[int]]: 
        if self.nonrand_neg_sample_size == 0:
            return self.pos_interactions[index].uqif(), random.sample(range(self.item_count), self.rand_neg_sample_size)
        else:
            positive_interaction = self.pos_interactions[index].uqif()
            negative_items_for_uq = self.neg_items_for_user_query_pair[(positive_interaction[0], positive_interaction[1])]
            if len(negative_items_for_uq) < self.nonrand_neg_sample_size: 
                random_sample_count = self.neg_sample_size - len(negative_items_for_uq)
                negative_items = random.sample(range(self.item_count), random_sample_count) + negative_items_for_uq
            else:
                negative_items = random.sample(negative_items_for_uq, self.nonrand_neg_sample_size)
                negative_items += random.sample(range(self.item_count), self.rand_neg_sample_size)
            return positive_interaction, negative_items
            
    def __init__(self, 
        fn_graph_info: str,
        fn_queries_multihot: str,
        fn_train_data: str,
        graph_type: type,
        random_negative_sample_size: int,
        non_random_negative_sample_size: int,
        device: torch.device):

        '''从几个数据文件中读取数据，初始化一个数据集类。'''

        super().__init__()
        assert graph_type in [Pps2DGraph, PpsHyperGraph, PpsLogHyperGraph], f'不支持的图类型：{str(graph_type)}'

        GraphDataset.device = device
        self.graph_type = graph_type
        self.rand_neg_sample_size = random_negative_sample_size
        self.nonrand_neg_sample_size = non_random_negative_sample_size
        self.neg_sample_size = random_negative_sample_size + non_random_negative_sample_size

        # 读取 graph_info.txt
        # 该文件中仅存储了四个值，如下所示
        with open(fn_graph_info, 'r', encoding='utf-8') as f:
            self.user_count, self.query_count, self.item_count, self.vocab_size = [int(part) for part in f.readline().strip().split()]
            self.node_count = self.user_count + self.query_count + self.item_count
            self.query_start_index_in_graph = self.user_count
            self.item_start_index_in_graph = self.user_count + self.query_count

        # 数据集中存储的并不是实体的 onehot 值，而是实体的索引
        # 例如 0,1,2 3 4 表示第 0 个 user、第 1 个 query、第 2 3 4 个 item
        # 在使用 embedding 层时，我们打算将 0 留作 padding_idx
        # 所以约定：实体的 onehot 值 == 实体的索引值 + 1
        self.users_onehot = th.tensor(range(1, 1 + self.user_count), device=device)
        self.items_onehot = th.tensor(range(1, 1 + self.item_count), device=device)
        self.vocabulary_onehot = th.tensor(range(1, 1 + self.vocab_size), device=device)

        # 这段代码构造 queries
        # 原来是使用稀疏矩阵表示所有的 queries
        # 后来发现 EmbeddingBag 更简单易用，就改用了后者，但仍然保留 queries_multihot

        indicesX, indicesY, elements = [], [], []
        embed_bag_input = []
        embed_bag_offsets = []
        embed_bag_offset = 0
        with open(fn_queries_multihot) as f_query:
            # 行形式举例：3294 3948 21039 343
            for row, line in enumerate(f_query):
                word_indices = [int(part) for part in line.strip().split()]
                word_onehots = [part + 1 for part in word_indices]
                embed_bag_offsets.append(embed_bag_offset)
                embed_bag_input.extend(word_onehots)
                embed_bag_offset += len(word_onehots)
                for word_index in word_indices:
                    indicesX.append(row)
                    indicesY.append(word_index)
                    elements.append(1 / len(word_indices))

        self.queries_multihot = torch.sparse_coo_tensor(
            [indicesX, indicesY], 
            elements, 
            (self.query_count, self.vocab_size), 
            dtype=torch.float, device=device
        ).coalesce()

        self.queries_for_embeddingbag = th.tensor(embed_bag_input, device=device)
        self.queries_offset_for_embeddingbag = th.tensor(embed_bag_offsets, device=device)
        
        # 读取 search logs
        self.search_logs = SearchLogCollection.read(fn_train_data)

        # 根据 search logs 整理出所有的正负交互
        self.pos_interactions = []
        neg_interactions = []
        neg_items_for_uq: Dict[Tuple[int, int], List[int]] = {}
        positive_flag_count = 0
        negative_flag_count = 0
        user_history_lens = {u:0 for u in range(self.user_count)}
        for log in self.search_logs:
            # 暂时不用高于1的相关度
            self.pos_interactions.extend(PosInteraction.from_search_log(log, treat_all_1=True))
            neg_items = neg_items_for_uq.setdefault((log.user, log.query), [])
            for item, flag in zip(log.items, log.interactions): 
                if flag > 0: 
                    positive_flag_count += 1
                    user_history_lens[log.user] += 1
                else: 
                    negative_flag_count += 1
                    neg_interactions.append((log.user, log.query, item))
                    neg_items.append(item)
        
        self.neg_interactions = neg_interactions
        self.neg_items_for_user_query_pair = neg_items_for_uq
        
        IOHelper.LogPrint(f'训练数据集构造完毕：{fn_train_data}')
        IOHelper.LogPrint(f'UserCount QueryCount ItemCount Vocabulary SearchLogs PosInteractions GraphType')
        IOHelper.LogPrint(f'{self.user_count:<9} {self.query_count:<10} {self.item_count:<9} {self.vocab_size:<10} ' + 
            f'{len(self.search_logs):<10} {len(self.pos_interactions):<15} {graph_type.__name__:<9}')
        IOHelper.LogPrint(f'每个正样本平均有 {negative_flag_count / positive_flag_count:<.4f} 个负样本')
        
        max_len = max(user_history_lens.values())
        if Gs.Dataset.user_history_limit != -1 and max_len > Gs.Dataset.user_history_limit:
            info = f'，因过长而被限制到 {Gs.Dataset.user_history_limit}'
        else:
            info = ''
        IOHelper.LogPrint(f'最大用户历史长度为 {max_len}{info}')
    
    def _get_user_history(self) -> None:
        qhis = [[] for _ in range(self.user_count)]
        ihis = [[] for _ in range(self.user_count)]
        for log in self.search_logs:
            qhis[log.user].append((log.search_time, log.query))
            ihis[log.user].extend(((t, i) for t, i, f in zip(log.times, log.items, log.interactions) if f > 0))
        for i in range(self.user_count):
            qhis[i] = [q for _, q in sorted(qhis[i])]
            ihis[i] = [item for _, item in sorted(ihis[i])]
            if Gs.Dataset.user_history_limit != -1:
                if len(qhis[i]) > Gs.Dataset.user_history_limit:
                    qhis[i] = qhis[i][:Gs.Dataset.user_history_limit]
                if len(ihis[i]) > Gs.Dataset.user_history_limit:
                    ihis[i] = ihis[i][:Gs.Dataset.user_history_limit]
        self._u_his_q = qhis
        self._u_his_i = ihis
        self._u_his_q_maxlen = len(max(qhis, key=lambda l: len(l)))
        self._u_his_i_maxlen = len(max(ihis, key=lambda l: len(l)))
    
    def _get_user_history_tensor(self) -> None:

        # 补足占位符
        qhis, ihis = self.user_history_queries, self.user_history_items
        user_queries_list, user_items_list = [], []
        for u in range(self.user_count):
            # 这里不能用 extend，否则就是修改数据集了
            user_queries_list.append(qhis[u] + [-1] * (self.user_history_query_maxlen - len(qhis[u])))
            user_items_list.append(ihis[u] + [-1] * (self.user_history_item_maxlen - len(ihis[u])))

        # 构造张量
        self._u_his_q_tensor = torch.tensor(user_queries_list, device=GraphDataset.device)
        self._u_his_i_tensor = torch.tensor(user_items_list, device=GraphDataset.device)
    
    @staticmethod
    def collate_fn(data: List[Tuple[Tuple[int, int, int, int], List[int]]]
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        '''返回值：正交互和负交互的四要素（用户索引，查询索引，商品索引，交互值），共八项（规定）。'''

        # Positive ones and negative ones
        p_users, p_queries, p_items, p_flags = [], [], [], []
        n_users, n_queries, n_items = [], [], []

        for (u, q, p_item, p_flag), neg_sample_items in data:
            
            l = len(neg_sample_items)
            n_items.extend(neg_sample_items)
            n_users.extend([u] * l)
            n_queries.extend([q] * l)

            p_users.append(u)
            p_queries.append(q)
            p_items.append(p_item)
            p_flags.append(p_flag)
        
        d = GraphDataset.device
        p_users   = th.tensor(p_users, device=d)
        p_queries = th.tensor(p_queries, device=d)
        p_items   = th.tensor(p_items, device=d)
        p_flags   = th.tensor(p_flags, device=d)

        n_users   = th.tensor(n_users, device=d)
        n_queries = th.tensor(n_queries, device=d)
        n_items   = th.tensor(n_items, device=d)
        n_flags   = th.zeros_like(n_users, device=d)

        return p_users, p_queries, p_items, p_flags, n_users, n_queries, n_items, n_flags



class TestSearchLogDataLoader(object):

    logs: List[Tuple[int, int, List[int], List[int], bool]]

    def __init__(self, fn_search_log: str, dataset_train: GraphDataset, device: torch.device):

        logs = []
        line_count = 0
        with open(fn_search_log, 'r', encoding='utf-8') as f:
            f.readline()
            for line in f:
                line_count += 1
                log = SearchLog.parse(line)
                # 暂时不用高于 1 的相关度
                tuple = log.get_interacted_items()
                if sum(log.interactions) > 0:
                    logs.append((log.user, log.query, tuple[0], None, True))
        
        self.logs = logs
        self.users1 = torch.ones(dataset_train.item_count, dtype=torch.long, device=device)
        self.queries1 = torch.ones(dataset_train.item_count, dtype=torch.long, device=device)

        IOHelper.LogPrint(f'验证/测试数据集构造完毕：{fn_search_log}')
        IOHelper.LogPrint(f'共 {line_count} 行，取出 {len(logs)} 条有效的 search_log')
    
    def __len__(self): return len(self.logs)

    def __iter__(self):
        for log in self.logs:
            u, q, items_interacted, flags_interacted, all_1 = log
            users = u * self.users1
            queries = q * self.queries1
            yield users, queries, items_interacted, flags_interacted, all_1
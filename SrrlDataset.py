from typing import Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import random, math

from Helpers.Torches import *
from Dataset import GraphDataset

class MetaPaths:

    positive_interactions: List[Tuple[int, int, int]]

    positive_heads:   Dict[Tuple[int, int], List[int]]
    positive_queries: Dict[Tuple[int, int], List[int]]
    positive_tails:   Dict[Tuple[int, int], List[int]]

    negative_heads:   Dict[Tuple[int, int], List[int]]
    negative_queries: Dict[Tuple[int, int], List[int]]
    negative_tails:   Dict[Tuple[int, int], List[int]]

    def __init__(self, graph_dataset: GraphDataset) -> None:

        self.graph_dataset = graph_dataset
        self.positive_interactions = []

        positive_heads:   Dict[Tuple[int, int], List[int]] = {}
        positive_queries: Dict[Tuple[int, int], List[int]] = {}
        positive_tails:   Dict[Tuple[int, int], List[int]] = {}

        negative_heads :  Dict[Tuple[int, int], List[int]] = {}
        negative_queries: Dict[Tuple[int, int], List[int]] = {}
        negative_tails :  Dict[Tuple[int, int], List[int]] = {}

        for pos_interaction in graph_dataset.pos_interactions:
            u, q, item, _ = pos_interaction.uqif()
            self.positive_interactions.append((u, q, item))
            uq = (u, q)
            qi = (q, item)
            ui = (u, item)
            
            if uq in positive_tails:   positive_tails[uq].append(item)
            else:                      positive_tails[uq] = [item]
            if qi in positive_heads:   positive_heads[qi].append(u)
            else:                      positive_heads[qi] = [u]
            if ui in positive_queries: positive_queries[ui].append(q)
            else:                      positive_queries[ui] = [q]
        
        for u, q, item in graph_dataset.neg_interactions:
            uq = (u, q)
            qi = (q, item)
            ui = (u, item)
            
            if uq in negative_tails:   negative_tails[uq].append(item)
            else:                      negative_tails[uq] = [item]
            if qi in negative_heads:   negative_heads[qi].append(u)
            else:                      negative_heads[qi] = [u]
            if ui in negative_queries: negative_queries[ui].append(q)
            else:                      negative_queries[ui] = [q]


        for uq, tails   in positive_tails.items():   positive_tails[uq]   = list(set(tails))
        for qi, heads   in positive_heads.items():   positive_heads[qi]   = list(set(heads))
        for ui, queries in positive_queries.items(): positive_queries[ui] = list(set(queries))

        for uq, tails   in negative_tails.items():   negative_tails[uq]   = list(set(tails))
        for qi, heads   in negative_heads.items():   negative_heads[qi]   = list(set(heads))
        for ui, queries in negative_queries.items(): negative_queries[ui] = list(set(queries))

        self.positive_heads = positive_heads
        self.positive_queries = positive_queries
        self.positive_tails = positive_tails

        self.negative_heads = negative_heads
        self.negative_queries = negative_queries
        self.negative_tails = negative_tails


class SrrlDatasetKG(Dataset):

    head_query_frequency: Dict[Tuple[int, int], int]

    def __len__(self): return len(self.meta_paths.positive_interactions)

    def __init__(self, 
        meta_paths: MetaPaths, 
        negative_sample_size: int, 
        mode: str = 'tail_batch',
        only_use_random_negative_sample: bool = True) -> None:
        super().__init__()
        self.meta_paths = meta_paths
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.only_random_sample = only_use_random_negative_sample
        self.head_query_frequency = dict()

        for u, q, _ in meta_paths.positive_interactions:
            head_query = (u, q)
            if head_query in self.head_query_frequency.keys(): self.head_query_frequency[head_query] += 1
            else: self.head_query_frequency[head_query] = 4


    def __getitem__(self, idx: int):
        meta_paths = self.meta_paths
        positive_sample = meta_paths.positive_interactions[idx]
        head, query, tail = positive_sample

        subsampling_weight = self.head_query_frequency[(head, query)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if self.only_random_sample:
            negative_sample = np.random.randint(self.meta_paths.graph_dataset.item_count, size=self.negative_sample_size)

            # negative_sample_list = []
            # negative_sample_size = 0

            # while negative_sample_size < self.negative_sample_size:
            #     negative_sample = np.random.randint(self.meta_paths.graph_dataset.ItemCount, size=self.negative_sample_size*2)
            #     if self.is_mode_head_batch():
            #         mask = np.in1d(
            #             negative_sample, 
            #             self.meta_paths.positive_heads[(query, tail)], 
            #             assume_unique=True, 
            #             invert=True
            #         )
            #     elif self.is_mode_company_batch():
            #         mask = np.in1d(
            #             negative_sample, 
            #             self.meta_paths.positive_tails[(head, query)], 
            #             assume_unique=True, 
            #             invert=True
            #         )
            #     else:
            #         raise ValueError('Training batch mode %s not supported' % self.mode)
            #     negative_sample = negative_sample[mask]
            #     negative_sample_list.append(negative_sample)
            #     negative_sample_size += negative_sample.size
            
            # if self.is_mode_head_batch() and (query, tail) in self.meta_paths.negative_heads.keys():
            #     negative_sample_list.append(self.meta_paths.negative_heads[(query, tail)])
            # elif self.is_mode_company_batch() and (head, query) in self.meta_paths.negative_tails.keys():
            #     negative_sample_list.append(self.meta_paths.negative_tails[(head, query)])
            # #negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            # negative_sample = np.random.choice(np.concatenate(negative_sample_list), size=self.negative_sample_size, replace=False)
        
        else:
            if self.is_mode_head_batch() and (query, tail) in self.meta_paths.negative_heads.keys(): 
                negative_sample = meta_paths.negative_heads[(query, tail)]
            elif self.is_mode_company_batch() and (head, query) in self.meta_paths.negative_tails.keys():
                negative_sample = meta_paths.negative_tails[(head, query)]
            else:
                raise ValueError(f'Training batch mode {self.mode} not supported')
            
            if len(negative_sample) > self.negative_sample_size:
                negative_sample = random.sample(negative_sample, self.negative_sample_size)
            elif len(negative_sample) < self.negative_sample_size:
                negative_sample += random.choices(negative_sample, k=(self.negative_sample_size - len(negative_sample)))

        if len(meta_paths.positive_tails[(head, query)]) > 0: 
            true_tail_company = LongTensor([random.choice(meta_paths.positive_tails[(head, query)])])
        else: 
            true_tail_company = LongTensor([tail])
        if len(meta_paths.positive_heads[(query, tail)]) > 0: 
            true_head_company = LongTensor([random.choice(meta_paths.positive_heads[(query, tail)])])
        else: 
            true_head_company = LongTensor([head])
        if len(meta_paths.positive_queries[(head, tail)]) > 0: 
            true_query_company = LongTensor([random.choice(meta_paths.positive_queries[(head, tail)])])
        else: 
            true_query_company = LongTensor([query])

        return (LongTensor(positive_sample), LongTensor(negative_sample), 
            subsampling_weight, self.mode, true_tail_company, true_head_company, true_query_company)

    def is_mode_head_batch(self) -> bool: return self.mode == 'head-batch'
    def is_mode_company_batch(self) -> bool: return self.mode == 'tail-company-batch' or self.mode == 'head-company-batch' or self.mode == 'query-company-batch'

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        true_tail_company = torch.stack([_[4] for _ in data], dim=0)
        true_head_company = torch.stack([_[5] for _ in data], dim=0)
        true_query_company = torch.stack([_[6] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight, mode, true_tail_company, true_head_company, true_query_company
    

class OneShotIterator:

    def __init__(self, 
        dataloader_tail_company:  DataLoader, 
        dataloader_head_company:  DataLoader, 
        dataloader_query_company: DataLoader):

        self.step = 0
        self.iterators = [
            self.one_shot_iterator(dataloader_tail_company),
            self.one_shot_iterator(dataloader_head_company),
            self.one_shot_iterator(dataloader_query_company)
        ]
        
    def next(self) -> Tuple[Tensor, Tensor, Tensor, str, Tensor, Tensor, Tensor]:
        it = self.iterators[self.step % 3]
        self.step += 1
        return next(it)
    
    def one_shot_iterator(self, dataloader):
        '''Transform a PyTorch Dataloader into python iterator.'''
        while True:
            for data in dataloader:
                yield data

from typing import List, Dict, Optional, Set, Any, Tuple, Type
import torch as torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor, LongTensor
import torch_sparse as thsp

from Dataset import GraphDataset
from Helpers.GlobalSettings import Gs, Gsv

class EmbeddingLayer(nn.Module):
    
    def __init__(self, 
        dataset: GraphDataset,
        embedding_size: int):
        super().__init__()

        # 记录参数
        self.dataset            = dataset
        self.embedding_size     = embedding_size

        self.users              = dataset.users_onehot
        self.queries            = dataset.queries_multihot
        self.queries_bag        = dataset.queries_for_embeddingbag
        self.queries_bag_offset = dataset.queries_offset_for_embeddingbag
        self.items              = dataset.items_onehot
        self.vocabulary         = dataset.vocabulary_onehot

        # self.qsp = thsp.SparseTensor.from_torch_sparse_coo_tensor(self.queries).coalesce()
        # assert self.qsp.device().type == self.device.type

        # Embedding 层
        self.embedding_user = EmbeddingLayer.create_embedding(len(self.users) + 1, self.embedding_size, padding_idx=0)
        self.embedding_item = EmbeddingLayer.create_embedding(len(self.items) + 1, self.embedding_size, padding_idx=0)
        self.embedding_bag_vocabulary = EmbeddingLayer.create_embedding_bag(len(self.vocabulary) + 1, self.embedding_size)
        # self.embedding_vocabulary = NNHelper.create_embedding(len(self.vocabulary) + 1, self.embedding_size)

        if Gs.Query.transform == Gsv.mean:
            pass
        elif Gs.Query.transform == Gsv.activation:
            self.query_transform = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                Gs.Query.transform_activation()
            )
        elif Gs.Query.transform == Gsv.rnn:
            raise NotImplementedError()
        else:
            raise ValueError()


    def forward(self, 
        user_indices:  LongTensor = None, 
        query_indices: LongTensor = None, 
        item_indices:  LongTensor = None):
        
        if Gs.Debug._calculate_embedding_info:
            Gs.Debug.embedding_info = (
                self.embedding_user.weight.abs().mean().item(),
                self.embedding_bag_vocabulary.weight.abs().mean().item(),
                self.embedding_item.weight.abs().mean().item(),
            )

        return (
            self.embed_user(user_indices),
            self.embed_query(query_indices),
            self.embed_item(item_indices)
        )
    

    def embed_user(self, user_indices: LongTensor = None) -> Tensor:
        return self.embedding_user(self.users if user_indices is None else (user_indices + 1))

    def embed_item(self, item_indices: LongTensor = None) -> Tensor:
        return self.embedding_item(self.items if item_indices is None else (item_indices + 1))

    def embed_query(self, query_indices: LongTensor = None) -> Tensor:

        # 采用 EmbeddingBag 的方案
        queries_embedded = self.embedding_bag_vocabulary(self.queries_bag, self.queries_bag_offset)
        if query_indices is not None:
            queries_embedded = queries_embedded[query_indices]
        
        if Gs.Query.transform == Gsv.activation:
            queries_embedded = self.query_transform(queries_embedded)

        # 采用子图的方案
        # vocabulary_embedded = self.embedding_vocabulary(self.vocabulary)
        # q = self.qsp[query_indices.to(self.device)]
        # queries_embedded = thsp.matmul(q, vocabulary_embedded)

        return queries_embedded

    @staticmethod
    def create_embedding(num_embeddings: int, embedding_dimension: int, padding_idx: int = None) -> nn.Embedding:
        emb = nn.Embedding(num_embeddings, embedding_dimension, padding_idx=padding_idx)
        init.xavier_uniform_(emb.weight)
        return emb
    

    @staticmethod
    def create_embedding_bag(num_embeddings: int, embedding_dimension: int, mode: str = 'mean') -> nn.Embedding:
        emb = nn.EmbeddingBag(num_embeddings, embedding_dimension, mode=mode)
        init.xavier_uniform_(emb.weight)
        return emb
        
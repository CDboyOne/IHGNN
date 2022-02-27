from typing import Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable

from Helpers.Torches import *
from Helpers.GlobalSettings import Gs, Gsv

class HemPredictionLayer(nn.Module):

    def __init__(self, 
        feature_dimension: int,
        lambda_muq:        float,
        item_count:        int):

        super().__init__()
        self.feature_dimension = feature_dimension
        self.lambda_muq = lambda_muq

        self.items_bias = Parameter(Tensor(item_count))
        init.normal_(self.items_bias)


    def forward(self, 
        user_feature:  Optional[Tensor], 
        query_feature: Tensor, 
        item_feature:  Tensor, 
        item_indices:  Optional[Tensor] = None):

        '''user_feature == None 说明不进行个性化；\n
        item_indices == None 说明在全体 item 上做预测。'''
        
        if item_indices is None: items_bias = self.items_bias
        else: items_bias = self.items_bias[item_indices]

        # 计算分数
        if user_feature is not None:
            m_uq = self.lambda_muq * query_feature + (1 - self.lambda_muq) * user_feature
        else:
            m_uq = query_feature
        if Gs.Prediction.use_cosine_similarity:
            similarity = torch.cosine_similarity(item_feature, m_uq)
            similarity += items_bias
        else:
            similarity = item_feature * m_uq
            similarity = similarity.sum(1) + items_bias
        return similarity


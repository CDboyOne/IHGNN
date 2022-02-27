from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable, Callable
import math
import torch
from torch import Tensor
from io import UnsupportedOperation


class Metrics:

    NDCG_at10: float
    HitRatio_at10: float
    MAP_at10: float

    title = 'HitRatio@10 NDCG@10 MAP@10'

    _idcg_dict: Dict[int, float] = dict()

    def __init__(self):
        self.NDCG_at10 = 0.0
        self.HitRatio_at10 = 0.0
        self.MAP_at10 = 0.0
    def add_to_self(self, m: 'Metrics') -> None:
        self.NDCG_at10 += m.NDCG_at10
        self.HitRatio_at10 += m.HitRatio_at10
        self.MAP_at10 += m.MAP_at10
    def divide_and_get_new(self, metrics_count: int) -> 'Metrics':
        m = Metrics()
        m.NDCG_at10 = self.NDCG_at10 / metrics_count
        m.HitRatio_at10 = self.HitRatio_at10 / metrics_count
        m.MAP_at10 = self.MAP_at10 / metrics_count
        return m

    def to_string(self, highlight: bool = False, no_title: bool = False) -> str:
        if no_title:
            return f'{self.HitRatio_at10:.4f} {self.NDCG_at10:.4f} {self.MAP_at10:.4f}'
        if highlight:
            line2 = f'\033[0;41m{self.HitRatio_at10:<11.4f} {self.NDCG_at10:<7.4f} {self.MAP_at10:<6.4f}\033[0m'
        else:
            line2 = f'{self.HitRatio_at10:<11.4f} {self.NDCG_at10:<7.4f} {self.MAP_at10:<6.4f}'
        return self.title + '\n' + line2
    def to_highlight_string(self) -> str: return self.to_string(highlight=True)
    def __str__(self):  return self.to_string()
    def __repr__(self): return self.to_string()


    @staticmethod
    def calculate_on_all_items(model_outputs: Tensor, interacted_items: List[int], flags: List[int], flags_are_all_1: bool) -> 'Metrics':

        '''在某次搜索中，计算评价指标。

        参数：
            model_outputs:    模型的输出。
            interacted_items: 列表，包含本次搜索中发生交互的 item 的索引。不能为空列表，否则会报错。
            flags:            列表，与 interacted_items 一一对应，表示对应 item 的相关性系数，为 1 或更高。
            flags_are_all_1:  表示 flags 中的元素是否全为 1。此参数为 True 时，flags 可为 None。

        返回值：评价指标对象。
        '''

        _, recommend_indices = torch.sort(model_outputs, descending=True)
        recommend_indices = recommend_indices[:10].cpu()
        indices_hit = []
        interacted_item_count_in_10 = (len(interacted_items) if len(interacted_items) < 10 else 10)

        # 计算所有预测对的 item 的索引（从 0 开始）
        if flags_are_all_1:
            for interacted_item in interacted_items:
                hit = torch.nonzero(recommend_indices == interacted_item, as_tuple=True)[0]
                if len(hit) > 0: indices_hit.append(hit.item())
        else:
            flags_hit = []
            for interacted_item, flag in zip(interacted_items, flags):
                hit = torch.nonzero(recommend_indices == interacted_item, as_tuple=True)[0]
                if len(hit) > 0: 
                    indices_hit.append(hit.item())
                    flags_hit.append(flag)
            flags_descending = sorted(flags_hit, reverse=True)

        m = Metrics()
        m.HitRatio_at10 = len(indices_hit) / interacted_item_count_in_10
        m.MAP_at10 = Metrics._get_map_for_all1(indices_hit)

        if flags_are_all_1:
            m.NDCG_at10 = Metrics._get_dcg_for_all1(indices_hit) / Metrics._get_idcg_for_all1(interacted_item_count_in_10)
        else:
            m.NDCG_at10 = Metrics._get_dcg(indices_hit, flags_hit) / Metrics._get_idcg(flags_descending)

        return m

    @staticmethod
    def _get_dcg(indices_hit: List[int], flags_hit: List[int]) -> float: return sum([math.log(2, i+2) * (2**r - 1) for i, r in zip(indices_hit, flags_hit)])
    @staticmethod
    def _get_dcg_for_all1(indices_hit: List[int]) -> float: return sum([math.log(2, i + 2) for i in indices_hit])
    @staticmethod
    def _get_idcg(flags_descending: List[int]) -> float: return sum([math.log(2, i+2) * (2**r - 1) for i, r in enumerate(flags_descending)])
    @staticmethod
    def _get_idcg_for_all1(truth_count: int) -> float: 
        idcg = Metrics._idcg_dict.get(truth_count, None)
        if idcg: return idcg
        else:
            idcg = sum([math.log(2, r_plus_1) for r_plus_1 in range(2, 2 + truth_count)])
            Metrics._idcg_dict[truth_count] = idcg
            return idcg
    @staticmethod
    def _get_map_for_all1(indices_hit: List[int]) -> float:
        l = len(indices_hit)
        if l == 0: return 0
        s = sum((j / (i+1) for i, j in zip(indices_hit, range(1, l+1))))
        return s / l


class MetricsCollection:

    _epochs: List[int]
    _tests: List[Metrics]
    _valids: List[Metrics]
    _has_valid: bool

    @property
    def has_valid(self) -> bool: return self._has_valid

    def __init__(self, has_valid: bool = False) -> None:
        self._has_valid = has_valid
        self._epochs, self._tests, self._valids = [], [], []
    
    def add(self, epoch: int, m_test: Metrics, m_valid: Metrics = None) -> None:
        if self.has_valid:
            if m_valid is None:
                raise ValueError('has_valid is True.')
            self._valids.append(m_valid)
        else:
            if m_valid is not None:
                raise ValueError('has_valid is False.')
        self._epochs.append(epoch)
        self._tests.append(m_test)
    
    def get_valid_best(self, key: Callable[[Metrics], Any], max_is_best: bool = True) -> Tuple[int, Metrics, Metrics]:
        '''return (epoch, test_metrics, valid_metrics).\n
        Example: key=lambda m: m.NDCG_at5'''
        if not self.has_valid:
            raise UnsupportedOperation('has_valid is False.')
        f = (max if max_is_best else min)
        i = self._valids.index(f(self._valids, key=key))
        return (self._epochs[i], self._tests[i], self._valids[i])
    
    def get_test_best(self, key: Callable[[Metrics], Any], max_is_best: bool = True
        ) -> Union[Tuple[int, Metrics, Metrics], Tuple[int, Metrics]]:
        '''return (epoch, test_metrics, valid_metrics) or (epoch, test_metrics) when has_valid==False.\n
        Example: key=lambda m: m.NDCG_at5'''
        f = (max if max_is_best else min)
        i = self._tests.index(f(self._tests, key=key))
        if self.has_valid:
            return (self._epochs[i], self._tests[i], self._valids[i])
        else:
            return (self._epochs[i], self._tests[i])
    
    def iter_epoch_test_valid(self) -> Iterable[Tuple[int, Metrics, Metrics]]:
        if not self.has_valid:
            raise UnsupportedOperation('has_valid is False.')
        return zip(self._epochs, self._tests, self._valids)
    def iter_epoch_test(self) -> Iterable[Tuple[int, Metrics]]:
        return zip(self._epochs, self._tests)


if __name__ == '__main__':

    model_outputs = [0.15, 0.05, 0.25, 0.05, 0.05, 0.13, 0.08, 0.12, 0.05, 0.07]
    ground_truth = [0, 7, 9]
    flags = [1, 1, 2]

    recommend_result = [_[1] for _ in sorted(zip(model_outputs, range(len(model_outputs))), reverse=True)]
    indices_hit = [i for i, r in enumerate(recommend_result) if r in ground_truth]
    res = Metrics.calculate_on_all_items(torch.Tensor(model_outputs), ground_truth, flags, True)

    print(f'Recommend Result: {recommend_result}')
    print(f'Hit indices (0 start): {indices_hit}')

    print(res)
    print(Metrics._get_idcg_for_all1(3))
    print(Metrics._get_idcg(sorted(flags, reverse=True)))

    m1 = res
    m2 = m1.divide_and_get_new(0.5)
    m3 = m1.divide_and_get_new(2)
    c = MetricsCollection(True)
    c.add(10, m1, m1)
    c.add(20, m2, m2)
    c.add(30, m3, m3)
    print(c.get_valid_best(key=lambda m: m.NDCG_at10))
    for e, mt, mv in c.iter_epoch_test_valid():
        print(e)
        print(mt)
        print(mv)
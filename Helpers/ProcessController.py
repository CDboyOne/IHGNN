from typing import Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import math

class ProcessController:

    StartEpoch: int
    CurrentEpoch: int
    EndEpoch: int
    EpochCount: int

    _start_test_epoch: int
    _test_frequency: int

    _start_store_epoch: Optional[int]
    _store_frequency: Optional[int]

    _test_count: int
    _train_time_list: List[float]
    _test_time_list: List[float]

    def __init__(self, 
        epoch_count: int, 
        start_epoch: int, 
        start_test_epoch: int, 
        test_frequency: int,
        start_store_epoch: int = None,
        store_frequency: int = None) -> None:

        '''初始化一个流程控制器。

        参数：
            epoch_count: ...
            start_epoch: 初始 epoch，从 1 开始。
            start_test_epoch: 开始测试的 epoch，从 1 开始。
            test_frequency: 控制每多少个 epoch 测试一次。
            start_store_epoch: 开始存档的 epoch，从 1 开始。为 None 则不存档。
            store_frequency: 控制每多少个 epoch 存档一次。为 None 则不存档。
        '''

        self.StartEpoch = start_epoch
        self.EpochCount = epoch_count
        self.EndEpoch = start_epoch + epoch_count
        self._start_test_epoch = start_test_epoch
        self._test_frequency = test_frequency
        self._test_count = 1 + (epoch_count - start_test_epoch) / test_frequency
        self._train_time_list = []
        self._test_time_list = []

        if start_store_epoch is None or store_frequency is None:
            self._start_store_epoch = self._store_frequency = None
        else:
            self._start_store_epoch = start_store_epoch
            self._store_frequency = store_frequency
    
    def __len__(self) -> int: return self.EpochCount
    
    def __iter__(self) -> Iterator[int]:
        self.CurrentEpoch = self.StartEpoch - 1
        return self
    
    def __next__(self) -> int:
        self.CurrentEpoch += 1
        if self.CurrentEpoch == self.EndEpoch: raise StopIteration()
        else: return self.CurrentEpoch
    
    def ShouldTest(self) -> bool:
        epoch = self.CurrentEpoch + 1
        start_test = self._start_test_epoch
        return (epoch - self.StartEpoch >= start_test) and ((self.CurrentEpoch - start_test) % self._test_frequency == 0 or (epoch == self.EndEpoch))
    
    def ShouldStore(self) -> bool:
        if self._start_store_epoch is None: return False
        epoch = self.CurrentEpoch + 1
        start_store = self._start_store_epoch
        return (epoch - self.StartEpoch >= start_store) and ((self.CurrentEpoch - start_store) % self._store_frequency == 0 or epoch == self.EndEpoch)
    
    def AddTrainTime(self, time: float) -> None: self._train_time_list.append(time)

    def AddTestTime(self, time: float) -> None:  self._test_time_list.append(time)

    def GetRemainingTime(self) -> float:
        if len(self._train_time_list) >= 2: 
            avg_epoch_time = (self._train_time_list[-1] + self._train_time_list[-2]) / 2
        elif len(self._train_time_list) == 1:
            avg_epoch_time = self._train_time_list[0]
        else:
            return float('nan')

        if len(self._test_time_list) >= 2:
            avg_test_time = (self._test_time_list[-1] + self._test_time_list[-2]) / 2
        elif len(self._test_time_list) == 1:
            avg_test_time = self._test_time_list[0]
        else:
            avg_test_time = avg_epoch_time * 2
        
        remain_epoch_time =  avg_epoch_time * (self.EndEpoch - self.CurrentEpoch)
        remain_test_time = avg_test_time * (self._test_count - len(self._test_time_list))
        return remain_epoch_time + remain_test_time
        
    def GetRemainingTimeString(self) -> float:
        time = self.GetRemainingTime()
        if math.isnan(time):
            return '暂无法计算'
        elif time >= 3600:
            h = time // 3600
            m = time / 60 - 60 * h
            return f'{int(h)} h {int(m)} m'
        elif time >= 60:
            return f'{int(time/60)} m'
        else:
            return f'{int(time)} s'


if __name__ == '__main__':
    pc = ProcessController(20, 5, 7, 2)
    for epoch in pc:
        print(f'Epoch {epoch}')
        if pc.ShouldTest(): print('  Test!')
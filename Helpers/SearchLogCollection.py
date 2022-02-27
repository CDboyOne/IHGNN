from typing import Iterable, Iterator, List, Dict, Set, Any, Tuple, Optional
import random, math

from Helpers.SearchLog import SearchLog, RawSearchLog

class SearchLogCollection:

    logs: List[SearchLog]

    def __init__(self, logs: List[SearchLog] = []): self.logs = list(logs)
    def __getitem__(self, index: int) -> SearchLog: return self.logs[index]
    def __len__(self) -> int: return len(self.logs)
    def __iter__(self) -> Iterator[SearchLog]: return iter(self.logs)

    def append(self, log: SearchLog): self.logs.append(log)

    def write(self, filename: str, encoding: str = 'utf-8') -> None:
        with open(filename, 'w', encoding=encoding) as fout:
            fout.write(SearchLog.column_names())
            fout.write('\n')
            for log in self.logs:
                fout.write(log.tostr())
                fout.write('\n')

    @classmethod
    def read(cls, filename: str, encoding: str = 'utf-8'):
        res = SearchLogCollection()
        with open(filename, 'r', encoding=encoding) as f: 
            f.readline()
            for line in f: 
                res.append(SearchLog.parse(line))
        return res


class RawSearchLogCollection:

    logs: List[RawSearchLog]
    _sorted: bool = False

    def __init__(self, search_logs: Iterable[RawSearchLog] = []): self.logs = list(search_logs)
    def __getitem__(self, index: int) -> RawSearchLog: return self.logs[index]
    def __len__(self) -> int: return len(self.logs)
    def __iter__(self) -> Iterator[RawSearchLog]: return iter(self.logs)

    def append(self, search_log: RawSearchLog): self.logs.append(search_log)
    def sort_by_user_and_time(self) -> None: 
        self.logs.sort(key=lambda log: (log.user_id, log.search_time))
        self._sorted = True

    def write(self, filename: str, encoding: str = 'utf-8') -> None:
        with open(filename, 'w', encoding=encoding) as fout:
            fout.write(RawSearchLog.column_names())
            fout.write('\n')
            for log in self.logs:
                fout.write(log.tostr())
                fout.write('\n')
    
    def write_search_ids(self, filename: str, encoding: str = 'utf-8') -> None:
        with open(filename, 'w', encoding=encoding) as fout:
            for log in self.logs:
                fout.write(log.search_id)
                fout.write('\n')
    

    def split_to_train_valid_test(self, 
        train_ratio: float, 
        valid_ratio: float, 
        test_ratio: float, 
        presplit_search_ids: Optional[List[Set[str]]], 
        reserve_at_least_one_in_train: bool = False
        ) -> Tuple['RawSearchLogCollection', 'RawSearchLogCollection', 'RawSearchLogCollection']:

        '''分割数据集。

        参数：
            reserve_at_least_one_in_train: 对每个 user，至少在训练集中保留一条 log。注意此项为 True 时可能破坏分割率。

        返回值：
            由训练集、验证集和测试集组成的三元组。

        分割方法：
            对于每个 user, 设其拥有 m 条 search log, 那么先取其前面 [m * train_ratio] 条 log，其中方括号表示向下取整；
            再以 {m * train_ratio} 的概率再取一条 log，其中花括号表示小数部分；
            这两部分 log 组合起来，一起进入训练集。
            再以相同方法，取尾部的相应数量的 log 进入测试集。
            中间剩下的 log 就是验证集了。
        '''

        if not self._sorted: self.sort_by_user_and_time()

        train_logs = RawSearchLogCollection()
        valid_logs = RawSearchLogCollection()
        test_logs = RawSearchLogCollection()

        if presplit_search_ids is not None:
            ids1, ids2, ids3 = presplit_search_ids
            for log in self.logs:
                if log.search_id in ids1:
                    train_logs.append(log)
                elif log.search_id in ids2:
                    valid_logs.append(log)
                elif log.search_id in ids3:
                    test_logs.append(log)
                else:
                    raise ValueError(f'search id {log.search_id} 不在任意一个预分割集合内。')
        else:
            logs = self.logs
            log_count = len(logs)
            # 起始索引（包含）和结束索引（不包含）
            user_start_index = 0
            user_valid_start_index = 0
            user_test_start_index = 0
            user_end_index = 1

            while user_end_index != (log_count + 1):

                # 当起止对应的 user 不同时，即定出了同一 user 的 log 范围
                if (user_end_index == log_count) or (logs[user_start_index].user_id != logs[user_end_index].user_id):

                    user_log_count = user_end_index - user_start_index

                    # 定出训练集
                    # 小数和整数部分
                    probability, train_log_count = math.modf(train_ratio * user_log_count)
                    train_log_count = int(train_log_count)
                    # 按概率决定是否要多取一个
                    train_log_count += random.choices([1, 0], weights=[probability, 1 - probability], k=1)[0]
                    # 决定是否要在训练集中至少保留一个
                    if reserve_at_least_one_in_train and (train_log_count == 0): train_log_count = 1
                    # 这就定出了训练集的部分
                    user_valid_start_index = user_start_index + train_log_count

                    # 定出测试集
                    # 如果训练集就把这些 log 占完了，就不用再搞了
                    if user_valid_start_index == user_end_index:
                        user_test_start_index = user_end_index
                    else:
                        probability, test_log_count = math.modf(test_ratio * user_log_count)
                        test_log_count = int(test_log_count)
                        test_log_count += random.choices([1, 0], weights=[probability, 1 - probability], k=1)[0]
                        user_test_start_index = user_end_index - test_log_count

                        if user_test_start_index < user_valid_start_index:
                            user_test_start_index = user_valid_start_index
                    
                    # 把对应的索引分别加进训练和测试集合
                    for index in range(user_start_index, user_valid_start_index): train_logs.append(logs[index])
                    for index in range(user_valid_start_index, user_test_start_index): valid_logs.append(logs[index])
                    for index in range(user_test_start_index, user_end_index): test_logs.append(logs[index])

                    # 该处理下一个 user 了
                    user_start_index = user_end_index

                user_end_index += 1
        
        return (train_logs, valid_logs, test_logs)
    

    def split_to_train_test(self, 
        test_data_ratio: float, 
        reserve_at_least_one_in_train: bool = False
        ) -> Tuple['RawSearchLogCollection', 'RawSearchLogCollection']:

        '''分割数据集。\n

        参数：\n
        test_data_ratio: 测试集所占比例，如 0.2 表示训练和测试集各占 80% 和 20%.\n
        reserve_at_least_one_in_train: 对每个 user，至少在训练集中保留一条 log。注意此项为 True 时可能破坏分割率。\n

        返回值：\n
        由训练集和测试集组成的二元组。\n

        分割方法：\n
        对于每个 user, 设其拥有 m 条 search_log. 那么先取其最后 [m * test_data_ratio] 条 log，其中方括号表示向下取整；\n
        再以 {m * test_data_ratio} 的概率再取一条 log，其中花括号表示小数部分；\n
        这些 log 组合起来，就得到测试集。
        '''

        if not self._sorted: self.sort_by_user_and_time()

        train_logs = RawSearchLogCollection()
        test_logs = RawSearchLogCollection()
        logs = self.logs
        log_count = len(logs)

        # 起始索引（包含）和结束索引（不包含）
        user_start_index = 0
        user_end_index = 1

        while user_end_index != log_count:

            # 当起止对应的 user 不同时，即定出了同一 user 的 log 范围
            if logs[user_start_index].user_id != logs[user_end_index].user_id:
                user_log_count = user_end_index - user_start_index
                # 小数和整数部分
                decimal, test_log_count = math.modf(test_data_ratio * user_log_count)
                test_log_count = int(test_log_count)
                # 按概率决定是否要多取一个
                test_log_count += random.choices([1, 0], weights=[decimal, 1 - decimal], k=1)[0]
                middle_index = user_end_index - test_log_count

                # 决定是否要在训练集中保留一个
                if reserve_at_least_one_in_train and (middle_index == user_start_index): middle_index += 1

                # 把对应的索引分别加进训练和测试集合
                for index in range(user_start_index, middle_index): train_logs.append(logs[index])
                for index in range(middle_index, user_end_index): test_logs.append(logs[index])

                # 该处理下一个 user 了
                user_start_index = user_end_index

            user_end_index += 1
        
        return (train_logs, test_logs)

    
    def to_onehot(self, 
        user_id_onehot_dict: Dict[str, int], 
        item_id_onehot_dict: Dict[str, int],
        query_rdict: Dict[str, int]) -> SearchLogCollection:

        res = SearchLogCollection()
        for log in self.logs:
            if not log.sorted:
                log.sort_items()
            u = user_id_onehot_dict[log.user_id]
            q = query_rdict[log.query]
            items = [item_id_onehot_dict[id] for id in log.item_ids]
            flags = log.interactions.copy()
            log2 = SearchLog(u, q, log.search_time, items, log.pages, log.positions, flags, log.times)
            res.append(log2)
        return res

    @classmethod
    def read(cls, filename: str, encoding: str = 'utf-8'):
        '''从文件中读取。'''
        logs = cls()
        with open(filename, 'r', encoding=encoding) as f:
            f.readline()
            for line in f:
                logs.append(RawSearchLog.parse(line))
        return logs
    

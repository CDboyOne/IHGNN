from typing import Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable, NamedTuple

class SearchLog(NamedTuple):

    user: int
    query: int
    search_time: str

    items: List[int] = None
    pages: List[int] = None
    positions: List[int] = None
    interactions: List[int] = None
    times: List[str] = None

    def tostr(self) -> str: 
        columns = [
            str(self.user),
            str(self.query),
            self.search_time,
            ' '.join([str(e) for e in self.items]),
            ' '.join([str(e) for e in self.pages]),
            ' '.join([str(e) for e in self.positions]),
            ' '.join([str(e) for e in self.interactions]),
            ' '.join(self.times)
        ]
        return ','.join(columns)
    def tostrs(self):
        for i in range(len(self.items)):
            cols = [self.user, self.query, self.search_time, 
                self.items[i], self.pages[i], self.positions[i], self.interactions[i], self.times[i]]
            yield ','.join((str(e) for e in cols))
    __str__ = tostr

    def get_interacted_items(self, flag_policy = 'min') -> Tuple[List[int], List[int], bool]:
        '''获取该 search log 中的发生交互的 item 及其相关性指数。

        参数：
            flag_policy: 对于交互过多次的 item，其相关性指数如何取。可选：min, max。

        返回值：
            Part1 items: 发生交互的商品索引。
            Part2 flags: 对应的相关性指数，为 1 或者更高。
            Part3 all_positive_flags_are_1: 为 True 时，说明所有相关性指数都是 1，否则为 False.
        '''
        pos_items: Dict[int, List[int]] = {}
        for item, flag in zip(self.items, self.interactions):
            if flag > 0:
                pos_items.setdefault(item, []).append(flag)

        items, flags = [], []
        all_positive_flags_are_1 = True
        f = (min if flag_policy == 'min' else max)

        for item, all_flags in pos_items.items():
            items.append(item)
            flag = f(all_flags)
            if flag > 1:
                all_positive_flags_are_1 = False
            flags.append(flag)
            
        return (items, flags, all_positive_flags_are_1)

    @classmethod
    def parse(cls, s: str):
        u, q, stime, items, pages, positions, flags, times = s.strip().split(',')
        items = [int(e) for e in items.split()]
        pages = [int(e) for e in pages.split()]
        positions = [int(e) for e in positions.split()]
        flags = [int(e) for e in flags.split()]
        times = times.split()
        return SearchLog(int(u), int(q), stime, items, pages, positions, flags, times)
    
    @staticmethod
    def column_names() -> str:
        return 'user,query,search_time,items,pages,positions,interactions,times'
    

class RawSearchLog:

    search_id: str
    user_id: str
    query: str
    search_time: str

    sorted: bool = False

    item_ids: List[str]
    pages: List[int]
    positions: List[int]
    interactions: List[int]
    times: List[str]

    def __init__(self, search_id: str, user_id: str, query: str, search_time: str):
        self.search_id, self.user_id, self.query, self.search_time = \
            search_id, user_id, query, search_time
        self.item_ids, self.pages, self.positions, self.interactions, self.times = \
            [], [], [], [], []

    def __str__(self): return self.tostr()
    def __repr__(self): return f'RawSearchLog(id={self.search_id}, len={len(self)}, sorted={self.sorted})'
    def __len__(self): return len(self.item_ids)
    
    def copy(self):
        log2 = RawSearchLog(self.search_id, self.user_id, self.query, self.search_time)
        log2.item_ids, log2.pages, log2.positions, log2.interactions, log2.times = \
            self.item_ids.copy(), self.pages.copy(), self.positions.copy(), self.interactions.copy(), self.times.copy()
        log2.sorted = self.sorted
        return log2

    def add_item(self, item_id: str, page: int, position: int, interaction: int, interaction_time: str) -> None:
        self.item_ids.append(item_id)
        self.pages.append(page)
        self.positions.append(position)
        self.interactions.append(interaction)
        self.times.append(interaction_time)
        self.sorted = False
    
    def sort_items(self) -> None:
        if not self.sorted:
            abs_positions = [1000 * page + pos for page, pos in zip(self.pages, self.positions)]

            table = list(zip(self.item_ids, self.pages, self.positions, self.interactions, self.times))
            table = [e for _, e in sorted(zip(abs_positions, table))]

            self.item_ids = [e[0] for e in table]
            self.pages = [e[1] for e in table]
            self.positions = [e[2] for e in table]
            self.interactions = [e[3] for e in table]
            self.times = [e[4] for e in table]

            self.sorted = True
    
    def tostr(self) -> str:
        columns = [
            self.search_id,
            self.user_id,
            self.query,
            self.search_time,
            str(self.sorted),
            ' '.join(self.item_ids),
            ' '.join([str(e) for e in self.pages]),
            ' '.join([str(e) for e in self.positions]),
            ' '.join([str(e) for e in self.interactions]),
            ' '.join(self.times)
        ]
        return '\t'.join(columns)
    
    def subset(self, item_ids_subset: Set[str]):
        log2 = self.copy()
        i = 0
        while i < len(log2.item_ids):
            if log2.item_ids[i] in item_ids_subset:
                i += 1
            else:
                del log2.item_ids[i]
                del log2.pages[i]
                del log2.positions[i]
                del log2.interactions[i]
                del log2.times[i]
        return log2

    def validate_times(self) -> None:
        for flag, itime in zip(self.interactions, self.times):
            if flag > 0:
                assert bool(itime) and itime != 'NA', repr(self) + '\n' + '\n'.join(self.tostr().split('\t'))

    @classmethod
    def parse(cls, line: str):
        parts = [part.strip() for part in line.strip().split('\t')]
        sid, uid, q, stime, sort, items, pages, positions, flags, times = parts
        log = cls(sid, uid, q, stime)
        log.sorted = (sort.strip() == 'True')
        log.item_ids = items.split()
        log.pages = [int(e) for e in pages.split()]
        log.positions = [int(e) for e in positions.split()]
        log.interactions = [int(e) for e in flags.split()]
        log.times = times.split()
        return log

    @staticmethod
    def column_names() -> str:
        return 'search_id\tuser_id\tquery\tsearch_time\tsorted\titem_ids\tpages\tpositions\tinteractions\ttimes'


class PosInteraction(NamedTuple):

    user: int
    query: int
    search_time: str
    item: int
    page: int
    position: int
    interaction: int
    time: str
    
    def uqif(self) -> Tuple[int, int, int, int]: return self.user, self.query, self.item, self.interaction
    def uqift(self) -> Tuple[int, int, int, int, str]: return self.user, self.query, self.item, self.interaction, self.time

    @staticmethod
    def from_search_log(log: SearchLog, treat_all_1: bool) -> List['PosInteraction']:
        ret = []
        for item, page, pos, flag, time in zip(log.items, log.pages, log.positions, log.interactions, log.times):
            if flag > 0:
                if treat_all_1 and flag > 1:
                    flag = 1
                ret.append(PosInteraction(log.user, log.query, log.search_time, item, page, pos, flag, time))
        return ret

class CikmSearchLog:

    session_id: str
    search_id: str

    user_id: str

    is_query: bool
    query_str: str
    query_segment: List[str]
    category_id: str

    item_ids: List[str]
    item_interactions: List[int]
    # 长度为 25 的字符串
    item_times: List[str]

    time_frame: int
    search_date: str
    page_alive_time: int

    @classmethod
    def from_CIKM_line(cls, line: str, category_mode: bool):

        '''从 CIKM 的原始数据集的某一行，构造一个原始的 search log. 为无效行时返回 None.

        参数：
            line: 行字符串。
            query_less_mode: 为 True 时，只处理不含 query words 的行；为 False 时，只处理含 query words 的行。
        
        注：query mode 下 CategoryId 为 -1，query less mode 下 QueryWords 为仅由 category id 构成的单个词的句子。
        '''
        
        query_id, session_id, user_id, time_frame, page_alive_time, event_date, query_str, category_id, items, is_test = line.strip().split(';')
        assert query_str != '' or category_id != ''

        if user_id == 'NA' or items == '' or items == 'NA' or is_test == 'TRUE':
            return None
        if (not category_mode) and query_str == '': 
            return None
        if (category_mode) and (category_id == '' or query_str != ''):
            return None

        if category_mode:
            query_str = ''
            query_words = []
            is_query = False
        else: 
            query_words = query_str.split(',')
            query_str = ' '.join(query_words)
            category_id = ''
            is_query = True

        items = items.split(',')
        flags = [0] * len(items)
        times = ['NA'] * len(items)

        log = CikmSearchLog()
        log.session_id = session_id
        log.search_id = query_id
        log.user_id = user_id
        log.is_query = is_query
        log.query_str = query_str
        log.query_segment = query_words
        log.category_id = category_id
        log.item_ids = items
        log.item_interactions = flags
        log.item_times = times
        log.time_frame = int(time_frame)
        log.search_date = event_date
        log.page_alive_time = page_alive_time
        return log
    
    def to_query(self, query_vocabulary_size: int) -> None:
        '''把类别浏览转换为查询。'''
        if not self.is_query:
            word = int(self.category_id) + query_vocabulary_size
            self.query_str = str(word)
            self.query_segment = [self.query_str]
            self.is_query = True

    def to_raw_search_log(self) -> RawSearchLog:
        log = RawSearchLog(self.search_id, self.user_id, self.query_str, self.search_date + str(self.time_frame).rjust(15, '0'))
        log.item_ids = self.item_ids
        log.pages = [1] * len(log.item_ids)
        log.positions = list(range(len(log.item_ids)))
        log.interactions = self.item_interactions
        log.times = self.item_times
        log.sorted = True
        return log
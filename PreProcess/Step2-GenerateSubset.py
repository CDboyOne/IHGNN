from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import os, sys, random, argparse
sys.dont_write_bytecode = True
sys.path.append('.')
os.umask(0)

from Helpers.PreProcessHelper import PreProcessHelper
from Helpers.IOHelper import IOHelper
from Helpers.SearchLog import RawSearchLog
from Helpers.SearchLogCollection import RawSearchLogCollection

# 决定是否生成针对 item 的 N-core 数据集。含义：从数据集中删去交互次数少于 N 的 item
process_item_N_core        = True
# 以下两个选项不能全为 True
# 决定是否生成针对 user 的 N-core 数据集。含义：从数据集中删去交互次数少于 N 的 user
process_user_N_core        = True
# 决定是否从所有 user 中随机选取 N_sample_user 个
process_user_random_sample = False

N_core_item = 5
N_core_user = 5
N_sample_user = 2_0000

source_folder = 'E:/DataScienceDataset/AlibabaAir/Intermediate/Complete5Core/'
result_folder = 'E:/DataScienceDataset/AlibabaAir/Intermediate/Subset02W5Core/'

source_folder = 'E:/DataScienceDataset/Cikm/Intermediate/WithCategory/'
result_folder = 'E:/DataScienceDataset/Cikm/Intermediate/WithCategory5Core/'

args = argparse.ArgumentParser()
args.add_argument('--source', default='', type=str, help='源数据目录')
args.add_argument('--result', default='', type=str, help='存储结果的目录')
args.add_argument('--nitem', default=0, type=int)
args.add_argument('--nuser', default=0, type=int)
args.add_argument('--rand_user', default=0, type=int)
args = args.parse_args()
source_folder = args.source or source_folder
result_folder = args.result or result_folder
if args.nitem:
    process_item_N_core = True
    N_core_item = args.nitem
if args.nuser:
    process_user_N_core = True
    process_user_random_sample = False
    N_core_user = args.nuser
elif args.rand_user:
    process_user_random_sample = True
    process_user_N_core = False
    N_sample_user = args.rand_user
else:
    if args.nitem:
        process_user_random_sample = False
        process_user_N_core = False

assert(source_folder != result_folder)
IOHelper.StartLogging(os.path.join(result_folder, 'PreProcess-Step2.txt'))
if not os.path.exists(result_folder): os.makedirs(result_folder)


# -------------------------
IOHelper.LogPrint()

item_ids = []
item_title_segments = []
vocabulary_item = set()

IOHelper.LogPrint('已有处理好的 item 数据，正在读取...')
item_ids: List[str]            = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'item_ids.txt'))
item_title_segments: List[str] = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'item_title_segments.txt'))
vocabulary_item: Set[str]  = set(IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'vocabulary_item.txt')))
IOHelper.LogPrint(f'完成，共 {len(item_ids)} 个 item。')

# -------------------------
IOHelper.LogPrint()

search_logs = RawSearchLogCollection()
user_ids = []
queries = []
query_segments = []
vocabulary_query = set()

IOHelper.LogPrint('已有处理好的 search 数据，正在读取...')
user_ids: List[str] = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'user_ids.txt'))
queries: List[str] = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'queries.txt'))
query_segments: List[str] = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'query_segments.txt'))
vocabulary_query: Set[str] = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'vocabulary_query.txt'))
search_logs = RawSearchLogCollection.read(os.path.join(source_folder, 'search_logs_raw.csv'))
IOHelper.LogPrint(f'完成，共 {len(search_logs)} 条 search log。')


# -------------------------
IOHelper.LogPrint()

IOHelper.LogPrint('正在采样子数据集...')

if not os.path.exists(result_folder): os.makedirs(result_folder)

item_ids_subset: Set[str] = set()
user_ids_subset: Set[str] = set()


# 产出一个集合 item_ids_subset
# 过滤集合 search_logs
if process_item_N_core:

    IOHelper.LogPrint(f'\n正在根据 item 的交互数量清洗数据...（最少 {N_core_item} 个）')

    # 记录所有 item 的交互数量
    item_interaction_count_dict: Dict[str, int] = {item_id : 0 for item_id in item_ids}
    for log in search_logs:
        for id, flag in zip(log.item_ids, log.interactions):
            if flag > 0:
                item_interaction_count_dict[id] += 1

    # 只保留交互数不少于 N 个的 item
    IOHelper.LogPrint(f'清洗前，每个 item 的平均交互数为：{sum(item_interaction_count_dict.values()) / len(item_interaction_count_dict)}')
    item_interaction_count_dict = {id : count for id, count in item_interaction_count_dict.items() if count >= N_core_item}
    IOHelper.LogPrint(f'清洗后，每个 item 的平均交互数为：{sum(item_interaction_count_dict.values()) / len(item_interaction_count_dict)}')
    item_ids_subset = set(item_interaction_count_dict.keys())

    # 过滤 search logs
    IOHelper.LogPrint('正在清洗 search logs...')
    logs_temp = RawSearchLogCollection()
    for log in search_logs:
        log2 = log.subset(item_ids_subset)
        if len(log2.item_ids) > 0:
            logs_temp.append(log2)
    search_logs = logs_temp


# 产出一个集合：
# user_ids_subset
if process_user_N_core:

    IOHelper.LogPrint(f'\n正在根据 user 的交互数量清洗数据...（最少 {N_core_user} 个）')

    # 记录所有 user 的交互数量
    user_interaction_count_dict: Dict[str, int] = {user_id : 0 for user_id in user_ids}
    for log in search_logs:
        for flag in log.interactions:
            if flag > 0:
                user_interaction_count_dict[log.user_id] += 1
    
    # 只保留交互数不少于 N 个的 user
    IOHelper.LogPrint(f'清洗前，每个 user 的平均交互数为：{sum(user_interaction_count_dict.values()) / len(user_interaction_count_dict)}')
    user_interaction_count_dict = {id : count for id, count in user_interaction_count_dict.items() if count >= N_core_user}
    IOHelper.LogPrint(f'清洗后，每个 user 的平均交互数为：{sum(user_interaction_count_dict.values()) / len(user_interaction_count_dict)}')
    user_ids_subset = set(user_interaction_count_dict.keys())


# 产出一个集合：
# user_ids_subset
if process_user_random_sample:
    IOHelper.LogPrint(f'\n正在随机选取 {N_sample_user} 个 user...')
    user_ids_subset = set(random.sample(user_ids, N_sample_user))


# 过滤集合 search_logs
if process_user_N_core or process_user_random_sample:
    logs_temp = RawSearchLogCollection()
    for log in search_logs:
        if log.user_id in user_ids_subset:
            logs_temp.append(log)
    search_logs = logs_temp


queries_segments_subset: Dict[str, str] = dict()
vocabulary_query_subset: Set[str] = set()
query_rdict = PreProcessHelper.GetReverseLookupDictionary(queries)

# 如果针对 user 进行了筛选，那么 item_ids_subset 必须被重新计算
if process_user_N_core or process_user_random_sample:
    item_ids_subset = set()

# 重建一些子集

for log in search_logs:
    if process_user_N_core or process_user_random_sample or (not process_item_N_core):
        item_ids_subset.update(log.item_ids)

    if (not process_user_N_core) and (not process_user_random_sample):
        user_ids_subset.add(log.user_id)

    if log.query not in queries_segments_subset.keys():
        query_segment = query_segments[query_rdict[log.query]]
        queries_segments_subset[log.query] = query_segment
        vocabulary_query_subset.update(query_segment.split())

item_title_segments_subset = []
vocabulary_item_subset: Set[str] = set()
item_ids_subset = list(item_ids_subset)
item_id_rdict = PreProcessHelper.GetReverseLookupDictionary(item_ids)
for id in item_ids_subset:
    segment = item_title_segments[item_id_rdict[id]]
    item_title_segments_subset.append(segment)
    vocabulary_item_subset.update(segment.split())

queries_subset = []
query_segments_subset = []
for query, segment in queries_segments_subset.items():
    queries_subset.append(query)
    query_segments_subset.append(segment)


IOHelper.LogPrint()
IOHelper.WriteListToFile(item_ids_subset, os.path.join(result_folder, 'item_ids.txt'))
IOHelper.WriteListToFile(item_title_segments_subset, os.path.join(result_folder, 'item_title_segments.txt'))
IOHelper.WriteListToFile(vocabulary_item_subset, os.path.join(result_folder, 'vocabulary_item.txt'))

search_logs.write(os.path.join(result_folder, 'search_logs_raw.csv'))
IOHelper.WriteListToFile(user_ids_subset, os.path.join(result_folder, 'user_ids.txt'))
IOHelper.WriteListToFile(queries_subset, os.path.join(result_folder, 'queries.txt'))
IOHelper.WriteListToFile(query_segments_subset, os.path.join(result_folder, 'query_segments.txt'))
IOHelper.WriteListToFile(vocabulary_query_subset, os.path.join(result_folder, 'vocabulary_query.txt'))

IOHelper.LogPrint(f'采样完毕，共 {len(search_logs)} 条 search log。')

IOHelper.EndLogging()
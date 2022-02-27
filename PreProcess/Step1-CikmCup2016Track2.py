from typing import Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import os, sys
sys.dont_write_bytecode = True
sys.path.append('.')

from Helpers.PreProcessHelper import PreProcessHelper
from Helpers.IOHelper import IOHelper
from Helpers.SearchLog import CikmSearchLog
from Helpers.SearchLogCollection import RawSearchLogCollection

# -------------------------
# 为 False 时只处理“搜索”，为 True 时处理“搜索”和“类别浏览”
with_category = True

dsdir = IOHelper.GetFirstLineContent('./dataset_dir.txt').strip()
dataset_queries = os.path.join(dsdir, 'Cikm/train-queries.csv')
dataset_items   = os.path.join(dsdir, 'Cikm/products.csv')
dataset_views   = os.path.join(dsdir, 'Cikm/train-item-views.csv')
dataset_clicks  = os.path.join(dsdir, 'Cikm/train-clicks.csv')

result_folder = f'{dsdir}/Cikm/Intermediate/WithCategory-WithView/'

IOHelper.StartLogging(os.path.join(result_folder, 'PreProcess-Cikm-Step1.txt'))

if not os.path.exists(dataset_queries): raise FileNotFoundError('文件不存在：', dataset_queries)
if not os.path.exists(dataset_items):   raise FileNotFoundError('文件不存在：', dataset_items)
if not os.path.exists(dataset_views):   raise FileNotFoundError('文件不存在：', dataset_views)
if not os.path.exists(dataset_clicks):  raise FileNotFoundError('文件不存在：', dataset_clicks)
if not os.path.exists(result_folder): os.makedirs(result_folder)

# ------------------------- 读取数据

raw_item_ids: List[str] = []
raw_item_prices: List[str] = []
raw_item_title_segments: List[str] = []
vocabulary_item: Set[str] = set()

IOHelper.LogPrint('\n读取 products.csv...')
with open(dataset_items, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        item_id, price, item_title_segment = line.strip().split(';')
        item_title_segment = item_title_segment.split(',')

        raw_item_ids.append(item_id)
        raw_item_prices.append(price)
        raw_item_title_segments.append(' '.join(item_title_segment))
        vocabulary_item.update(item_title_segment)

IOHelper.LogPrint(f'处理完毕，共 {len(raw_item_ids)} 个 item')

raw_item_id_rdict = PreProcessHelper.GetReverseLookupDictionary(raw_item_ids)

cikm_search_logs: List[CikmSearchLog] = []
vocabulary_query = set()
user_ids = set()
item_ids = set()
item_title_dict: Dict[str, str] = dict()
queries = set()
query_word_max = 0

IOHelper.LogPrint('\n读取 train-queries.csv...')
with open(dataset_queries, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        log = CikmSearchLog.from_CIKM_line(line, category_mode=False)
        if log is None: continue

        cikm_search_logs.append(log)
        vocabulary_query.update(log.query_segment)
        user_ids.add(log.user_id)
        item_ids.update(log.item_ids)
        queries.add(log.query_str)

if with_category:
    with open(dataset_queries, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            log = CikmSearchLog.from_CIKM_line(line, category_mode=True)
            if log is None: continue

            cikm_search_logs.append(log)
            user_ids.add(log.user_id)
            item_ids.update(log.item_ids)

user_ids = list(user_ids)
queries  = list(queries)
item_ids = list(item_ids)
item_prices = [raw_item_prices[raw_item_id_rdict[item_id]] for item_id in item_ids]
item_title_segments = [raw_item_title_segments[raw_item_id_rdict[item_id]] for item_id in item_ids]

# query_id, item_id, time(25)
click_tuples: List[Tuple[str, str, str]] = []

IOHelper.LogPrint('读取 train-clicks.csv...')
with open(dataset_clicks, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        query_id, timeframe, item_id = line.strip().split(';')
        click_tuples.append((query_id, item_id, '0000-00-00' + timeframe.rjust(15, '0')))

IOHelper.LogPrint('构建 query_id -> search_log 反查字典...')
log_query_rdict: Dict[str, CikmSearchLog] = {}
for log in cikm_search_logs:
    assert log.search_id not in log_query_rdict
    log_query_rdict[log.search_id] = log

# # search_id, user_id, item_id, time(25)
# view_tuples: List[Tuple[str, str, str, str]] = []

# IOHelper.LogPrint('读取 train-views.csv...')
# with open(dataset_views, 'r', encoding='utf-8') as f:
#     f.readline()
#     for l in f:
#         session_id, user_id, item_id, timeframe, event_date = l.strip().split(';')
#         if user_id == 'NA': continue
#         view_tuples.append((session_id, user_id, item_id, event_date + timeframe.rjust(15, '0')))

# IOHelper.LogPrint('构建 (session_id, user_id) -> search_logs 反查字典')
# log_session_rdict: Dict[Tuple[str, str], List[CikmSearchLog]] = {}
# for log in cikm_search_logs:
#     log_session_rdict.setdefault((log.session_id, log.user_id), []).append(log)

IOHelper.LogPrint('改写 search logs 中的 click 数据...')
click_count = 0
for query_id, item_id, itime in click_tuples:
    try:
        log = log_query_rdict.get(query_id, None)
        if log is None: continue

        i = log.item_ids.index(item_id)
        if log.item_interactions[i] == 0:
            log.item_interactions[i] = 1
            log.item_times[i] = itime
            click_count += 1
    except ValueError: pass

# IOHelper.LogPrint('改写 search logs 中的 view 数据...')
# view_count = 0
# view_count2 = 0
# for session_id, user_id, item_id, itime in view_tuples:
#         count_view_flag = True
#         logs = log_session_rdict.get((session_id, user_id), None)
#         if logs is None: continue

#         for log in logs:
#             try:
#                 i = log.item_ids.index(item_id)
#                 if log.item_interactions[i] == 0:
#                     log.item_interactions[i] = 1
#                     log.item_times[i] = itime
#                     view_count2 += 1
#                     if count_view_flag:
#                         view_count += 1
#                         count_view_flag = False
#             except ValueError:
#                 pass
                
IOHelper.LogPrint(f'有效的 click 共有 {click_count} 条。')
# IOHelper.LogPrint(f'有效的 view 共有 {view_count} 条，改写到数据上共 {view_count2} 条。')

if with_category:
    IOHelper.LogPrint('正在将类目浏览改写为搜索...')
    vocabulary = vocabulary_item | vocabulary_query
    max_token = 1 + max(map(int, vocabulary))
    new_vocabulary = set()
    for log in cikm_search_logs:
        if not log.is_query:
            log.to_query(max_token)
            new_vocabulary.add(log.query_str)
    new_vocabulary = list(new_vocabulary)
    vocabulary.update(new_vocabulary)
    vocabulary_query.update(new_vocabulary)
    queries.extend(new_vocabulary)


# ------------------------- 输出至文件

IOHelper.LogPrint('输出至文件...')
IOHelper.WriteListToFile(item_ids,            os.path.join(result_folder, 'item_ids.txt'))
IOHelper.WriteListToFile(item_prices,         os.path.join(result_folder, 'item_prices.txt'))
IOHelper.WriteListToFile(item_title_segments, os.path.join(result_folder, 'item_title_segments.txt'))
IOHelper.WriteListToFile(vocabulary_item,     os.path.join(result_folder, 'vocabulary_item.txt'))
IOHelper.WriteListToFile(vocabulary_query,    os.path.join(result_folder, 'vocabulary_query.txt'))
IOHelper.WriteListToFile(user_ids,            os.path.join(result_folder, 'user_ids.txt'))
IOHelper.WriteListToFile(queries,             os.path.join(result_folder, 'queries.txt'))
IOHelper.WriteListToFile(queries,             os.path.join(result_folder, 'query_segments.txt'))

raw_logs = RawSearchLogCollection((log.to_raw_search_log() for log in cikm_search_logs))
raw_logs.write(os.path.join(result_folder, 'search_logs_raw.csv'))
IOHelper.LogPrint(f'共 {len(raw_logs)} 条 search log，已写入文件 {result_folder}/search_logs_raw.csv')

IOHelper.EndLogging()
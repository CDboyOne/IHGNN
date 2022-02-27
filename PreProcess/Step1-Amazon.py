from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import os, sys, json, random, argparse
sys.dont_write_bytecode = True
sys.path.append('.')
os.umask(0)

from Helpers.PreProcessHelper import PreProcessHelper
from Helpers.IOHelper import IOHelper
from Helpers.SearchLog import RawSearchLog
from Helpers.SearchLogCollection import RawSearchLogCollection

base_dir       = 'E:/DataScienceDataset/Amazon/'
dataset_item   = base_dir + 'meta_CDs.json'
dataset_search = base_dir + 'CDs_5.json'
result_folder  = base_dir + 'CDs/Intermediate/'

args = argparse.ArgumentParser()
args.add_argument('--item', default='', type=str, help='商品信息文件的路径')
args.add_argument('--search', default='', type=str, help='搜索数据文件的路径')
args.add_argument('--result', default='', type=str, help='存储结果的目录')
args = args.parse_known_args()[0]
dataset_item   = args.item or dataset_item
dataset_search = args.search or dataset_search
result_folder  = args.result or result_folder

assert os.path.exists(dataset_item)
assert os.path.exists(dataset_search)
if not os.path.exists(result_folder): os.makedirs(result_folder)
IOHelper.StartLogging(os.path.join(result_folder, 'PreProcess-Step1.txt'))

IOHelper.LogPrint(f'\n正在处理 item data：{dataset_item}')

item_ids = set() # item_id 的集合
queries = set()  # query_str 的集合
vocabulary_query = set() # query_str 中的词构成的词典
# 从 item_id 到 Set[query_str] 的映射，表示某个 item 属于哪些 query
item_queries_dict: Dict[str, Set[str]] = {}

for index, line in enumerate(PreProcessHelper.yield_amazon_lines(dataset_item)):

    if index > 0 and index % 50000 == 0:
        IOHelper.LogPrint(f'处理到第 {index} 行...')

    line = json.loads(line)
    item_id = line['asin']
    categories = line['category']

    if len(categories) == 0: 
        continue

    categories = [PreProcessHelper.PreProcessText(c) for c in categories]
    query = ' '.join(categories)
    
    # 把 categories 按空格全部分割
    query_words = []
    for category in categories: 
        query_words.extend(category.split(' '))
    assert len(query_words) > 0

    item_ids.add(item_id)
    queries.add(query)
    vocabulary_query.update(query_words)

    item_queries_dict.setdefault(item_id, set()).add(query)


IOHelper.LogPrint(f'\n正在处理 review data: {dataset_search}')

# user_id 的集合
user_ids = set()
# 所有在 review dataset 中的 item_id 的集合
item_ids_useful = set()
# 从 user_id 到 List[Tuple[str, str]] 的映射，表示某个 user 都评论过哪些 item 以及评论时间
# Tuple[str, str] = Tuple[item, review_time]，如果多次评论过同一个 item，以最早一次时间为准
user_items_times_dict: Dict[str, List[Tuple[str, str]]] = {}

review_count = 0
invalid_review_count = 0

for index, line in enumerate(PreProcessHelper.yield_amazon_lines(dataset_search)):

    if index > 0 and index % 100000 == 0:
        IOHelper.LogPrint(f'处理到第 {index} 行...')

    line = json.loads(line)
    review_count += 1
    user_id = line['reviewerID']
    item_id = line['asin']
    review_time = str(line['unixReviewTime'])

    # 问题：为什么有的 item_id 会不在 metadata 里面？？？
    if item_id in item_ids:
        item_ids_useful.add(item_id)
    else:
        invalid_review_count += 1
        continue

    user_ids.add(user_id)
    user_items_times_dict.setdefault(user_id, []).append((item_id, review_time))
        
    
IOHelper.LogPrint(
    f'\n处理完毕，共有 {review_count} 条 review，{invalid_review_count} 条 review 的 item_id 无效。')
IOHelper.LogPrint(
    f'共有 {len(user_ids)} 个 user, {len(queries)} 个 query, {len(item_ids)} 个 item，' + 
    f'留下了在评论数据集中出现的 {len(item_ids_useful)} 个 item。\n')

# 及时释放内存，防止内存溢出

IOHelper.WriteListToFile(user_ids, os.path.join(result_folder, 'user_ids.txt'))
user_ids = None

item_ids = item_ids_useful
IOHelper.WriteListToFile(item_ids, os.path.join(result_folder, 'item_ids.txt'))
IOHelper.WriteListToFile(['<span'] * len(item_ids), os.path.join(result_folder, 'item_title_segments.txt'))
item_ids = None

queries = list(queries)
IOHelper.WriteListToFile(queries, os.path.join(result_folder, 'queries.txt'))
IOHelper.WriteListToFile(queries, os.path.join(result_folder, 'query_segments.txt'))
queries = None

IOHelper.WriteListToFile(set(['<span']), os.path.join(result_folder, 'vocabulary_item.txt'))
IOHelper.WriteListToFile(vocabulary_query, os.path.join(result_folder, 'vocabulary_query.txt'))

# 构造 search logs
search_logs = RawSearchLogCollection()
for i, (user_id, itemids_times_here) in enumerate(user_items_times_dict.items()):
    for item_id, review_time in itemids_times_here:
        for query in item_queries_dict[item_id]:
        # for query in random.sample(item_queries_dict[item_id], 1):
            log = RawSearchLog(str(i), user_id, query, review_time)
            log.add_item(item_id, 1, 1, 1, review_time)
            search_logs.append(log)

IOHelper.LogPrint('正在将原始 search log 写入文件...')
search_logs.write(os.path.join(result_folder, 'search_logs_raw.csv'))
IOHelper.LogPrint(f'共 {len(search_logs)} 条 search log，已写入文件 {result_folder}/search_logs_raw.csv')
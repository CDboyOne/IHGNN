from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable
import sys, os, argparse
sys.dont_write_bytecode = True
sys.path.append('.')
os.umask(0)

from Helpers.PreProcessHelper import PreProcessHelper
from Helpers.IOHelper import IOHelper
from Helpers.SearchLog import RawSearchLog, SearchLog
from Helpers.SearchLogCollection import RawSearchLogCollection, SearchLogCollection

def read_search_ids(filename: str) -> Set[str]:
    with open(filename, 'r', encoding='utf-8') as f:
        return set((l.strip() for l in f))

def get_interaction_count(logs: SearchLogCollection) -> Tuple[int, int]:
    positive_flag_count, negative_flag_count = 0, 0
    for log in logs:
        i = len([1 for flag in log.interactions if flag > 0])
        positive_flag_count += i
        negative_flag_count += len(log.interactions) - i
    return positive_flag_count, negative_flag_count

reserve_at_least_one_in_train = True

source_folder = 'E:/DataScienceDataset/AlibabaAir/Intermediate/Complete1Core/'
result_folder = 'E:/DataScienceDataset/AlibabaAir/Complete1Core2/'
split_ratios = [0.66, 0.1, 0.24]

source_folder = 'E:/DataScienceDataset/AlibabaAir/Intermediate/Complete5Core/'
result_folder = 'E:/DataScienceDataset/AlibabaAir/Complete5Core2/'
split_ratios = [0.695, 0.095, 0.21]

source_folder = 'E:/DataScienceDataset/AlibabaAir/Intermediate/Subset01W5Core/'
result_folder = 'E:/DataScienceDataset/AlibabaAir/Subset01W5Core/'
split_ratios = [0.696, 0.094, 0.21]

source_folder = 'E:/DataScienceDataset/Cikm/Intermediate/WithCategoryWithView5Core/'
result_folder = 'E:/DataScienceDataset/Cikm/WithCategoryWithView5Core/'
split_ratios = [0.69, 0.09, 0.22]

args = argparse.ArgumentParser()
args.add_argument('--source', default='', type=str, help='源数据目录')
args.add_argument('--result', default='', type=str, help='存储结果的目录')
args.add_argument('--split', default='', type=str, help='分割比例，可以只指定前两项，如：\'0.7, 0.1\'')
args.add_argument('--presplit', default='', type=str, help='指定预先分割好的 search_id 文件所在的目录')
args = args.parse_args()
source_folder = args.source or source_folder
result_folder = args.result or result_folder
if args.source or args.result:
    if args.split:
        split_ratios = [float(r.strip()) for r in str(args.split).split(',')]
        if len(split_ratios) == 2: split_ratios += [1 - sum(split_ratios)]
    else:
        split_ratios = [0.7, 0.1, 0.2]

assert(source_folder != result_folder)
IOHelper.StartLogging(os.path.join(result_folder, 'PreProcess-Step3.txt'))

# --------------------------------------------------
IOHelper.LogPrint('读取文件...')

item_ids       = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'item_ids.txt'))
queries        = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'queries.txt'))
item_segments  = [line.split() for line in IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'item_title_segments.txt'))]
query_segments = [line.split() for line in IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'query_segments.txt'))]
user_ids       = IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'user_ids.txt'))
vocabulary     = list(set(IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'vocabulary_item.txt'))) \
                   .union(IOHelper.ReadStringListFromFile(os.path.join(source_folder, 'vocabulary_query.txt'))))
search_logs    = RawSearchLogCollection.read(os.path.join(source_folder, 'search_logs_raw.csv'))

# --------------------------------------------------
IOHelper.LogPrint('读取完毕，开始生成反查字典...')
user_id_rdict    = PreProcessHelper.GetReverseLookupDictionary(user_ids)
query_rdict      = PreProcessHelper.GetReverseLookupDictionary(queries)
item_id_rdict    = PreProcessHelper.GetReverseLookupDictionary(item_ids)
vocabulary_rdict = PreProcessHelper.GetReverseLookupDictionary(vocabulary)

# --------------------------------------------------
IOHelper.LogPrint('\n开始分割数据集并将其转换为 one hot 形式...')

# 将 search logs 按 user 与 search time 排序
search_logs.sort_by_user_and_time()

# 分割 search logs
if args.presplit:
    search_idss = [
        read_search_ids(os.path.join(args.presplit, 'search_ids_train.txt')),
        read_search_ids(os.path.join(args.presplit, 'search_ids_valid.txt')),
        read_search_ids(os.path.join(args.presplit, 'search_ids_test.txt')),
    ]
    IOHelper.LogPrint(f'将使用以下目录中的预定义的分割指示文件：{args.presplit}')
else:
    search_idss = None
    IOHelper.LogPrint(f'定下的训练、验证、测试集的比例为：{split_ratios}')
    IOHelper.LogPrint(f'对特定 user，至少在训练集中保留一条：{reserve_at_least_one_in_train}')
train_logs, valid_logs, test_logs = search_logs.split_to_train_valid_test(
    *split_ratios, 
    presplit_search_ids=search_idss,
    reserve_at_least_one_in_train=reserve_at_least_one_in_train
)
IOHelper.LogPrint(f'search logs 已分割，原共 {len(search_logs)} 条，现训练、验证、测试集各' 
    + f' {len(train_logs)} {len(valid_logs)} {len(test_logs)} 条')
IOHelper.LogPrint(f'三者占比各为：{len(train_logs) / len(search_logs):.3f}' 
    + f' {len(valid_logs) / len(search_logs):.3f} {len(test_logs) / len(search_logs):.3f}')

# 为产生可复制的结果，将三个集合的 search id 写入文件
train_logs.write_search_ids(os.path.join(result_folder, 'search_ids_train.txt'))
valid_logs.write_search_ids(os.path.join(result_folder, 'search_ids_valid.txt'))
test_logs.write_search_ids(os.path.join(result_folder, 'search_ids_test.txt'))

train_logs = train_logs.to_onehot(user_id_rdict, item_id_rdict, query_rdict)
valid_logs = valid_logs.to_onehot(user_id_rdict, item_id_rdict, query_rdict)
test_logs  = test_logs.to_onehot(user_id_rdict, item_id_rdict, query_rdict)

# --------------------------------------------------
IOHelper.LogPrint('\n制作统计数据...')

IOHelper.LogPrint(f'UserCount QueryCount ItemCount Vocabulary')
IOHelper.LogPrint(f'{len(user_ids):<9} {len(queries):<10} {len(item_ids):<9} {len(vocabulary):<10}')

IOHelper.LogPrint(f'\n各数据集的 search log 数量：')
IOHelper.LogPrint(f'Train     Valid     Test     ')
IOHelper.LogPrint(f'{len(train_logs):<9} {len(valid_logs):<9} {len(test_logs):<9}')

node_count = len(user_ids) + len(queries) + len(vocabulary)
positive_flag_count, negative_flag_count = get_interaction_count(train_logs)
IOHelper.LogPrint(f'\n训练集中正负交互数量：{positive_flag_count} {negative_flag_count}')
IOHelper.LogPrint(f'平均每个正样本有 {negative_flag_count / positive_flag_count:.4f} 个负样本')
IOHelper.LogPrint(f'稀疏度（正交互数 / 用户查询商品数量之和的平方）为：{positive_flag_count / (node_count ** 2)}')

positive_flag_count += get_interaction_count(valid_logs)[0] + get_interaction_count(test_logs)[0]
IOHelper.LogPrint(f'\n全集中正交互数量：{positive_flag_count}')

# --------------------------------------------------
IOHelper.LogPrint('\n将 train valid test 数据写入文件...')
train_logs.write(os.path.join(result_folder, 'train_data.csv'))
valid_logs.write(os.path.join(result_folder, 'valid_data.csv'))
test_logs.write(os.path.join(result_folder, 'test_data.csv'))

# 将 vocabulary 写入文件
IOHelper.WriteListToFile(vocabulary, os.path.join(result_folder, 'vocabulary.txt'))

# 生成 item_titles_multihot.txt
with open(os.path.join(result_folder, 'item_titles_multihot.txt'), 'w', encoding='utf-8') as fout:
    for segments in item_segments:
        onehots = [vocabulary_rdict[segment] for segment in segments]
        fout.write(' '.join([str(onehot) for onehot in onehots]) + '\n')

# 生成 queries_multihot.txt
with open(os.path.join(result_folder, 'queries_multihot.txt'), 'w', encoding='utf-8') as fout:
    for segments in query_segments:
        onehots = [vocabulary_rdict[segment] for segment in segments]
        fout.write(' '.join([str(onehot) for onehot in onehots]) + '\n')

# 写入 user query item vocabulary 集合的大小
with open(os.path.join(result_folder, 'graph_info.txt'), 'w', encoding='utf-8') as fout:
    fout.write(f'{len(user_ids)} {len(queries)} {len(item_ids)} {len(vocabulary)}')

IOHelper.EndLogging()
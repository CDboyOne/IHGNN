import argparse
from Helpers.GlobalSettings import Gsv

class ConsoleArgs:

    checkpoint: str
    storecheckpoint: bool
    storemetrics: bool

    epoch_count: int
    epoch_start_test: int
    epoch_test_frequency: int
    dataset: str

    model: str
    gnn: str
    gnns: int
    feature_order: int

    completeness: str
    long_tail_filename: str

    device: str
    batch_size: int
    embedding_size: int

    def __init__(self, args: argparse.Namespace):

        self.checkpoint = args.checkpoint
        self.storecheckpoint = args.storecheckpoint
        self.storemetrics = args.storemetrics

        self.epoch_count = args.epoch_count
        self.epoch_start_test = args.epoch_start_test
        self.epoch_test_frequency = args.epoch_test_frequency
        self.dataset = args.dataset

        self.model = args.model
        self.gnn = args.gnn
        self.gnns = args.gnns
        self.feature_order = args.feature_order

        self.completeness = args.completeness
        self.long_tail_filename = args.longtail

        self.device = args.device
        self.embedding_size = args.embedding_size

def parse_args() -> ConsoleArgs:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', '--cp', default='', type=str,
        help='指定要加载的模型存档文件名，或者若为 latest 则加载最新的存档，若为空则不加载。')
    
    parser.add_argument('--storecheckpoint', '--scp', '-c', action='store_true', default=False,
        help='当指定此选项时，会在每次测试前存档模型。')

    parser.add_argument('--storemetrics', '--sm', '-m', action='store_true', default=False,
        help='当指定此选项时，会存档测试结果。')
    
    parser.add_argument('--epoch_count', '--ec', type=int, default=0,
        help='指定本次训练的 epoch 总数。默认从主脚本中读取。')

    parser.add_argument('--epoch_start_test', '--est', type=int, default=0,
        help='指定开始测试的 epoch 数。从 1 开始。默认从主脚本中读取。')

    parser.add_argument('--epoch_test_frequency', '--etf', type=int, default=0,
        help='指定测试频率（epoch 数）。从 1 开始。默认从主脚本中读取。')
    
    parser.add_argument('--dataset', '--ds', type=str, default='',
        help='指定本次训练或测试所用的数据集位置。默认从主脚本中读取。')
    
    parser.add_argument('--model', type=str, default='',
        help='指定所用模型。默认从主脚本中读取。')
    
    parser.add_argument('--gnn', type=str, default='',
        help='指定图神经网络层类型。默认从主脚本中读取。')
    
    parser.add_argument('--gnns', type=int, default=0,
        help='指定图神经网络层数。默认从主脚本中读取。')
    
    parser.add_argument('--feature_order', '--fo', type=int, default=0,
        help='指定特征交互阶数。默认从主脚本中读取。')
    
    parser.add_argument('--completeness', type=str, default=Gsv.graph_uqi,
        help=f'指定图的完整性。可选值：{Gsv.graph_uqi} {Gsv.graph_only_uq} {Gsv.graph_only_ui} {Gsv.graph_only_qi}。')
    
    parser.add_argument('--longtail', type=str, default='',
        help=f'指定长尾信息统计文件名。不提供表示不统计。')
    
    parser.add_argument('--device', '-d', type=str, default='',
        help='指定张量所在的设备，可为数字，例如 0 指代 cuda:0；cpu 指代 CPU。默认从主脚本中读取。')
    
    parser.add_argument('--embedding_size', '--emb', type=int, default=0,
        help='指定 embedding size。默认从主脚本中读取。')

    return ConsoleArgs(parser.parse_args())
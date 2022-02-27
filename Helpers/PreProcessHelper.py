from typing import Type, Any, List, Dict, Set, Tuple, Union, Optional, Iterator, Iterable, Callable, NamedTuple
import re, gzip
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

class PreProcessHelper:

    re_punc = re.compile(
        '[\s+\.\-\\\!\/_,$%^*(+\"\')]+|=[+——()?:;|【】“”！，。？、~@#￥%……&*（）]+')
    stop_words = set(stopwords.words('english')) | set(' ')
    stemmer_porter = SnowballStemmer("english")  

    @staticmethod
    def PreProcessText(text: str) -> str:
        """去掉字符串中没用的部分，并将其格式化。"""
        # 小写化，去除数字、标点、一些符号、末尾空格
        text = re.sub(r'\d+', '', text.lower())
        text = PreProcessHelper.re_punc.sub(' ', text).strip()

        # 去除stopwords
        tokens = word_tokenize(text)
        result = (PreProcessHelper.stemmer_porter.stem(t) 
            for t in tokens if t not in PreProcessHelper.stop_words)

        return ' '.join(result)


    @staticmethod
    def GetReverseLookupDictionary(list: List[Any], start_index: int = 0) -> Dict[Any, int]:
        """从某个集合构建其反查字典。默认的起始索引为 0。示例：\n
        ['abc', 'acb', 'bca']\n
        -->\n
        ['abc': 0, 'acb': 1, 'bca': 2]
        """
        return {item : (index + start_index) for index, item in enumerate(list)}
    
    
    @staticmethod
    def yield_amazon_lines(filename: str) -> Iterable[str]:

        '''传入 Amazon 数据集的文件名，返回一个迭代器，迭代的内容是文件每一行代表的 json 对象。
        支持 .json 和 .json.gz。'''

        if filename.endswith('.gz'):
            with gzip.open(filename, 'rb') as f:
                for l in f:
                    yield l.strip()
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                for l in f:
                    yield l.strip()
        else:
            raise ValueError(f'无效的扩展名。文件名为：{filename}')


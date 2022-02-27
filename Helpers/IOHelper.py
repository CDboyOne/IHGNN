from typing import Any, Iterable, List, Tuple, Dict, Set
import sys, codecs, os, time, re


class IOHelper:

    log_filename: str = None
    warned_about_cannot_log: bool = False
    color_code_re: str = '\033\\[0;*[0-9]*m'

    @staticmethod
    def GetAllContent(filename: str, encoding: str = 'utf-8') -> str:
        with open(filename, 'r', encoding=encoding) as f:
            return ''.join(f.readlines())
            
    @staticmethod
    def GetFirstLineContent(filename: str, encoding: str = 'utf-8') -> str:
        with open(filename, 'r', encoding=encoding) as f:
            return f.readline().strip()

    @staticmethod
    def CanLogToFile() -> bool: return IOHelper.log_filename != None

    @staticmethod
    def StartLogging(log_filename: str = None):
        if log_filename is not None:
            IOHelper.log_filename = log_filename
            folder, _ = os.path.split(log_filename)
            if not os.path.exists(folder): os.makedirs(folder)
            with open(IOHelper.log_filename, 'w', encoding='utf-8') as f:
                f.write('\n')
        else:
            IOHelper.warned_about_cannot_log = True
        if sys.platform == 'linux':
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        if log_filename is not None:
            IOHelper.LogPrint(f'开始向以下文件写入日志：{log_filename}')
        
        # 玄学：这样做才能让控制台高亮输出
        os.system('')

    @staticmethod
    def EndLogging():
        if IOHelper.CanLogToFile():
            IOHelper.LogPrint(f'以下文件的日志记录已结束：{IOHelper.log_filename}')
            IOHelper.log_filename = None

    @staticmethod
    def LogPrint(message_no_endline: str = '', put_time_in_single_line: bool = False):
        
        if message_no_endline != '':
            split_index = 0
            for i in range(len(message_no_endline)):
                if message_no_endline[i] != '\n':
                    split_index = i
                    break

            part1 = message_no_endline[:split_index]
            part2 = message_no_endline[split_index:]
            t = time.strftime('[%H:%M:%S] ', time.localtime())
            if put_time_in_single_line: t = t + '\n'
            message_no_endline = part1 + t + part2

        if IOHelper.CanLogToFile():
            with open(IOHelper.log_filename, 'a', encoding='utf-8') as f:
                # Remove the color code in message
                # 删除字符串中的颜色代码
                message_no_endline = str(message_no_endline.encode('utf-8'), encoding='utf-8')
                f.write(re.sub(IOHelper.color_code_re, '', message_no_endline))
                f.write('\n')
        else:
            if not IOHelper.warned_about_cannot_log:
                print('Warning: Please call IOHelper.StartLogging() first.')
                IOHelper.warned_about_cannot_log = True

        print(message_no_endline)
        sys.stdout.flush()

            
    
    @staticmethod
    def WriteListToFile(list : Iterable, filename : str):
        with open(filename, 'w', encoding='utf-8') as fout:
            for item in list:
                fout.write(str(item) + '\n')
        log_line = f'列表共包含 {len(list)} 个元素，已写入文件 {filename}'
        IOHelper.LogPrint(log_line)

    @staticmethod
    def ReadStringListFromFile(filename: str, encoding: str = 'utf-8') -> List[str]:
        res = list()
        with open(filename, 'r', encoding=encoding) as fin:
            for line in fin:
                res.append(line.strip())
        return res

import psutil 
import os

def print_memory_info():
    """
    Used for improving program
    """
    print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    info = psutil.virtual_memory()
    print( u'电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024) )
    print(u'当前使用的总内存占比：',info.percent)
    print(u'cpu个数：',psutil.cpu_count())

if __name__=='__main__':
    print_memory_info()
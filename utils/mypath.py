"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)
# 修改了下文件夹路径，原来的前3个全是/data/dzy/，最后一个是 /data/dzy/Imagenet/  试试看data/wnx 少一个前面的斜杠  data/wnx和./data/wnx的效果是一样的
        if database == 'cifar-10':
            return 'data/wnx/'
        
        elif database == 'cifar-20':
            return 'data/wnx/'   #后面几个全删了斜杠

        elif database == 'stl-10':
            return 'data/wnx/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return 'data/wnx/Imagenet/'
        
        else:
            raise NotImplementedError

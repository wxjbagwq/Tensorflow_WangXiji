# -*- coding: utf-8 -*-
import sys,os
sys.path.insert(0, src)                              # 导入本地py文件(src文件夹下面的几个.py文件)
import scipy.misc                                    # scipy负责导入图片，以及对图像进行处理
import numpy as np
from optimize import optimize                        # 前面这个optimize是/src/optimize.py，后面这个是这个文件里面的一个函数！
                                                     # optimize函数功能:迭代优化
from argparse as ArgumentParser                      # 解析命令行参数   
from utils import save_img, get_img, exists, list_files
import evaluate                                      # 这个应该就是当前文件夹下面的evalute.py

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT   = 1e2
TV_WEIGHT      = 2e2                                 # 权变差，为了让图像更加真实
                    
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATION = 2000
LEARNING_RATE  = 1e-3           
NUM_EPOCHS     = 2              
# epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。
# 简单说，epochs指的就是训练过程中数据将被“轮”多少次。
# 如果训练集有1000个样本，batchsize=10，那么训练完这个样本集需要100次iteration，1次epoch。
# 一个iteration等于使用一个batchsize样本训练一次。
VGG_PATH       = 'data/'






















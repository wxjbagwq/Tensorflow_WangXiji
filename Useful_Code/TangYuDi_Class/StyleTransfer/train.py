# -*- coding: utf-8 -*-
import sys,os
sys.path.insert(0, src)                                  # 导入本地py文件(src文件夹下面的几个.py文件)
import scipy.misc                                        # scipy负责导入图片，以及对图像进行处理
import numpy as np
from optimize import optimize                            # 前面这个optimize是/src/optimize.py，后面这个是这个文件里面的一个函数！
                                                         # optimize函数功能:迭代优化
from argparse import ArgumentParser                      # 解析命令行参数   
from utils import save_img, get_img, exists, list_files  # 这些都是自己编写的一些工具
import evaluate                                          # 这个应该就是当前文件夹下面的evalute.py

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT   = 1e2
TV_WEIGHT      = 2e2                                     # 权变差，为了让图像更加真实
                    
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_ITERATION = 2000
LEARNING_RATE  = 1e-3           
NUM_EPOCHS     = 2              
# epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。
# 简单说，epochs指的就是训练过程中数据将被“轮”多少次。
# 如果训练集有1000个样本，batchsize=10，那么训练完这个样本集需要100次iteration，1次epoch。
# 一个iteration等于使用一个batchsize样本训练一次。
VGG_PATH       = 'data/imagenet-vgg-verydeep-19.mat'
TRAIN_PATH     = 'data/train2014/'
BATCH_SIZE     = 4
DEVICE         = 'gpu:0'

# 解析命令行参数！ 
def build_parser():
    parser = ArgumentParser()                             
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=True)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test', type=str,
                        dest='test', help='test image path',
                        metavar='TEST', default=False)

    parser.add_argument('--test-dir', type=str,
                        dest='test_dir', help='test image save dir',
                        metavar='TEST_DIR', default=False)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS',
                        default=CHECKPOINT_ITERATIONS)

    parser.add_argument('--vgg-path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content-weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight',
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)

    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

# 检查命令的必要参数是否存在！
def check_opts(opts):

#   
  exists(opts.checkpoint_dir, "checkpoint dir not found!")
  exists(opts.style, "style path not found!")
  exists(opts.train_path, "train path not found!")
  if opts.test or opts.test_dir:
      exists(opts.test, "test img not found!")
      exists(opts.test_dir, "test directory not found!")
  exists(opts.vgg_path, "vgg network data not found!")

# assert作为一个断言函数，负责判断后面的语句是否为真，只有为真的时候才会继续执行 
  assert opts.epochs > 0
  assert opts.batch_size > 0
  assert opts.checkpoint_iterations > 0
  assert os.path.exists(opts.vgg_path)
  assert opts.content_weight >= 0
  assert opts.style_weight >= 0
  assert opts.tv_weight >= 0
  assert opts.learning_rate >= 0 
  
  
def main():
  parser  = build_parser()
  options = parser.parse_args()                                # 注意这里的引用关系，实际上是argparse.parse_args()！
  check_opts(options)
  
  style_target = get_img(options.style)                        # 这里就是用自己写的get_img去拿出参数里面的style图像！
  

if __name__ == '__main__':
 main()  






















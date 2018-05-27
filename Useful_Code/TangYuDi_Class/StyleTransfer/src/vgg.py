# -*- coding = utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io
import pdb

MEAN_PIXEL = [123.68, 116.779, 103.939]  # 这个值是模型提供者给出的图像均值数据！

# net函数就定义了这个VGG的卷积模型
def net(model_path, image):

# (1) 先定义一个层的结构layer，用'()'即元组表示；有16个卷积层
  layers = (
           'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
           
           'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
           
           'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
           'relu3_3', 'conv3_4', 'relu3_4', 'pool3'
           
           'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
           'relu4_3', 'conv4_4', 'relu4_4', 'pool4'
           
           'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
           'relu5_3', 'conv5_4', 'relu5_4'
           )
          
# (2) 处理模型参数数据，由于是.mat的模型，所以需要用scipy.io.loadmat读入         
  data = scipy.io.loadmat(model_path)                                    # 执行完毕后就拿出了模型文件的参数！                  
# (2.1) 从这里可以看出来这个模型文件实际上是一种字典结构！
  mean = data['normalization'][0][0][0] 
  mean_pixel = np.mean(mean, axis=(0, 1))                                # 这个用法是根据文档描述来的！
  weights = data['layers'][0]                                            # 先取出所有的参数
  
  net = {}                                                               # 模型
  current = input_image                                                  # 当前输入

# (2.2) 获取feature_map(FP)  
  for i,name in enumerate(layers):                                       # 见第3笔记本后136或者step
    kind = name[:4]                                                      # [:4]表示前4个元素
    if kind == 'conv':                                                   # conv层就取出w,b
      kernels, bias = weights[i][0][0][0][0]                             # (1) 这里的kernel表示权重；这里的weights的格式如(2)所诉
      kernels = np.transpose(kernels, (1,0,2,3))                         # (2) 对于'.mat'文件的格式是：W,H,in_channel，out_channel
      bias = bias.reshape(-1)                 # 见第3笔记本后137          #     而对于tf的格式：H,W,in_channel，out_channel
      current = _conv_layer(current, kernels, bias)                      # 让输入的img经过卷积
      
      


# VGG预处理模块，因为使用的别人训练好的VGG模型都进行了减均值的预处理操作，这里以相同方法也要实现。
def preprocess(image):                   
  return image - MEAN_PIXEL














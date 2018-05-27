# -*- coding: utf-8 -*-

# 这里非常重要！由于模型里面用的vgg实际上是pre-trained的！那么如何实际用上呢？
# 就需要在这个./src/vgg.py里面再把用到的vgg的整个模型再写一遍，然后导入权重等进去即可！
import vgg, pdb, time, os
import tensorflow as tf
import numpy as np
# 这个库也是自己编写的！
import transform 
from utils import get_img

# (1) 对于CNN来说，实际的feature_map输出一定是relu层的输出！因为必须要经过非线性函数才能使CNN有效！
# (2) 下面的层的选取是根据论文来的。
# (3) 拿出这些输出去计算损失。
STYLE_LAYERS = ('relu1_1','relu2_1','relu3_1','relu4_1','relu5_1')
CONTENT_LAYER = 'relu4_2'
# DEVICES = 'CUDA_VISIBLE_DEVICES'

def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
             
###             
  mod = len(content_targets) % batch_size       # 由于batch_size=4,如果content一共有7个样本，那么只能构成1个batch
  if mod > 0:                                   # 所以后面的3个就要被丢弃，这段代码就完成这样的功能！
    content_targets = content_targets[:-mod]    # *** 这里的[:-mod]表示取content从第1个开始到倒数第mod个样本！ ***
###  
  style_features = {}
  batch_shape = (batch_size, 256, 256, 3)       # batch_size * H * W * C；这是tensorflow的标准数据格式
  style_shape = (1,) + style_target.shape       # 前面的1表示style的batch_size = 1
  
  with tf.Graph.as_default(),tf.device('/gpu:0'),tf.Session as sess:      # 分别是指定：tf的计算图，运行设备，Session
    style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
    style_image_pre = vgg.preprocess(style_image)
    net = vgg.net(vgg_path, style_image_pre)                              # 调用来让预处理完成的style图片跑一遍VGG
    stylp_pre = np.array(style_target)
    for layer in STYLE_LAYERS:
      features = net[layer].eval(feed_dict = {style_image：stylp_pre})   # 取出对应层的feature_map
      features = np.reshape(features, (-1, features.shape[3]))           ###
      gram = np.matmul(features.T, features)                             # 这里的3行代码是根据论文的设计实现而已
      style_features[layer] = gram                                       ###
      
   with tf.Graph.as_default(),tf.Session as sess:                         # 分别是指定：tf的计算图，运行设备，Session   
     x_content = tf.placeholder(tf.float32, shape=batch_shape,name='x_content')
     x_pre = vgg.preprocess(x_content)
     content_features = {}
     content_net = vgg.net(vgg_path, x_pre)
     content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]
     
    preds = transform.net(x_content/255.0)
     
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      






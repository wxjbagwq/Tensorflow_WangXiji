# -*- coding: utf-8 -*-

# 这里非常重要！由于模型里面用的vgg实际上是pre-trained的！那么如何实际用上呢？
# 就需要在这个./src/vgg.py里面再把用到的vgg的整个模型再写一遍，然后导入权重等进去即可！
import vgg, pdb, time, os
import tensorflow as tf
import numpy as np
# 这个库也是自己编写的！
import transform 
from utils import get_img

def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
             
###             
  mod = len(content_targets) % batch_size       # 由于batch_size=4,如果content一共有7个样本，那么只能构成1个batch
  if mod > 0:                                   # 所以后面的3个就要被丢弃，这段代码就完成这样的功能！
    content_targets = content_targets[:-mod]    # *** 这里的[:-mod]表示取content从第1个开始到倒数第1个样本！ ***
###  
  style_features = []
  batch_shape = (batch_size, 256, 256, 3)       # batch_size * H * W * C；这是tensorflow的标准数据格式
  style_shape = (1,) + style_target.shape       # 前面的1表示style的batch_size = 1
  
  with tf.Graph.as_default(),tf.device('/gpu:0'),tf.Session as sess:      # 分别是指定：tf的计算图，运行设备，Session
    style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
    style_image_pre = vgg.preprocess(style_image)
    
  
  










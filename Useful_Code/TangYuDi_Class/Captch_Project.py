# -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
import random
number = ['0','1','2','3','4','5','6','7','8','9']
#alphabet = ['a','b'，'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#ALPHABET = ['A','B'，'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#print("number结构：", type(number))
#生成随机验证码的数字（注意这个时候还不是字符串形式！）
#例子中的定义法：def random_captcha_text(char_set=number, captcha_size=4):
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []                             # 这是一个空的list用来存放随机生成的字符串
    for i in range(captcha_size):                 # 这个范围表示了captcha_size来限定长度
        char = random.choice(char_set)            # 见笔记本
        captcha_text.append(char)                 # 见笔记本，这是一个list的核心操作：向list里添加元素，但是一次只能添加一个！
    return captcha_text
#根据字符串生成验证码的图像
def generate_random_captcha_text():
    image = ImageCaptcha()                        # 导入时：from captcha.image import ImageCaptcha
    
    captcha_text = random_captcha_text(number, 4)
    captcha_text = ''.join(captcha_text)          # 将list结构存的数字转化成为字符串！
    
    captcha = image.generate(captcha_text)        # 调用函数负责生成图像！
    #image.write(captcha_text, captcha_text + '.jpg')可以把生成的图片保存下来
    
    captcha_image = Image.open(captcha)           # 调用open函数打开图片，这个应该是要进行下一步操作的标准过程
    captcha_image = np.array(captcha_image)       # 见笔记本
    return captcha_text, captcha_image            # 返回的是标签label和输入! 其他见笔记本
#灰度图的转换
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面是简便方法，下面是正规办法
        # r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
def text2vec(text):  
    text_len = len(text)  
    if text_len > MAX_CAPTCHA:  
        raise ValueError('验证码最长4个字符')  
   
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)  
    """
    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  
    """
    for i, c in enumerate(text):  
        idx = i * CHAR_SET_LEN + int(c)  
        vector[idx] = 1  
    return vector  
# 向量转回文本  
def vec2text(vec):  
    """
    char_pos = vec.nonzero()[0]  
    text=[]  
    for i, c in enumerate(char_pos):  
        char_at_pos = i #c/63  
        char_idx = c % CHAR_SET_LEN  
        if char_idx < 10:  
            char_code = char_idx + ord('0')  
        elif char_idx <36:  
            char_code = char_idx - 10 + ord('A')  
        elif char_idx < 62:  
            char_code = char_idx-  36 + ord('a')  
        elif char_idx == 62:  
            char_code = ord('_')  
        else:  
            raise ValueError('error')  
        text.append(chr(char_code)) 
    """
    text=[]
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):  
        number = i % 10
        text.append(str(number)) 
             
    return "".join(text)  
   
""" 
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有 
vec = text2vec("F5Sd") 
text = vec2text(vec) 
print(text)  # F5Sd 
vec = text2vec("SFd5") 
text = vec2text(vec) 
print(text)  # SFd5 
"""  
#生成batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])       # 初始化好batch_x为0,包括格式
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])       # 初始化好batch_y为0,包括格式
    
    # 有时生成图像大小不是(60, 160, 3)
    def wrap_generate_captcha_text_and_image():                      # 这里没有大看懂
        while True:
            text, image = generate_random_captcha_text()
            if image.shape == (60, 160, 3):
                return text, image
    for i in range(batch_size):
        text, image = wrap_generate_captcha_text_and_image()
        image = convert2gray(image)                                  # 转化成为灰度图
        
        batch_x[i, :] = image.flatten()/255                          # 这里是把像素值全部除以255使之都是0`1之间的数
        batch_y[i, :] = text2vec(text)                               # 这里是把str的标签转化成40维的向量！
    return batch_x, batch_y
#正向传播
def fp_captcha_cnn(w_alpha, b_alpha):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])      # 这里的x是喂给CNN的！所以要符合tf的4D格式！用tf.reshape转换！
    
    #conv1
    w_c1  = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))     # filter的宽，高，深度，第四个参数是filter的个数！
    b_c1  = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, dropout_percent)
    
    #conv2
    w_c2  = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2  = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, dropout_percent)
       
    #conv3
    w_c3  = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3  = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, dropout_percent)
    
    #full_connection
    w_fc1 = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_fc1 = tf.Variable(b_alpha*tf.random_normal([1024]))
    fc1   = tf.reshape(conv3, [-1, w_fc1.get_shape().as_list()[0]])
    fc1   = tf.nn.relu(tf.add(tf.matmul(fc1, w_fc1), b_fc1))
    fc1   = tf.nn.dropout(fc1, dropout_percent)
    
    #output
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, CHAR_SET_LEN*MAX_CAPTCHA]))
    b_out = tf.Variable(b_alpha*tf.random_normal([CHAR_SET_LEN*MAX_CAPTCHA]))
    out   = tf.add(tf.matmul(fc1, w_out), b_out)
    
    return out
#搭建神经网络
def train_captcha_cnn():
    output    = fp_captcha_cnn(0.01, 0.1)
    loss      = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))  # 这里用的是交叉熵函数tf.nn.corss_entropy
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict   = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)                                            # 写0行不行？
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)     # 写0行不行？
    correct_pred = tf.equal(max_idx_p, max_idx_l) 
    accuracy  = tf.reduce_mean(tf.cast(correct_pred, tf.float32))                # 用的batch所以要求均值！batch的值在sess里面喂入！
    
    saver= tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())                              # 初始化变量
    
        step = 0                                                                 # 训练次数记录
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, dropout_percent: 0.75})  # 这里的符号没有看懂！
            print(step, loss_)
            
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_x_test, dropout_percent: 1.})
                print(step, acc)
                if acc > 0.85:
                    saver.save(sess, "./验证码识别/model/crack_captcha.model", global_step=step)
                    break
            step += 1
#测试网络
def crack_captcha(captcha_image):  
    output = crack_captcha_cnn()  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        saver.restore(sess, "./验证码识别/model/crack_captcha.model")                # 读取之前的网络
   
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})  
        text = text_list[0].tolist()  
        return text 
#主函数（通过train控制训练或检测）
if __name__ == '__main__':
    train = 0
    if train == 0:                                 # 训练
        number = ['0','1','2','3','4','5','6','7','8','9']
        #alphabet = ['a','b'，'c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        #ALPHABET = ['A','B'，'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        text,image = generate_random_captcha_text()
        print("验证码图像尺寸：",np.shape(image))  # 示例代码：print("验证码图像尺寸：",image.shape)
        IMAGE_HEIGHT = np.shape(image)[0]          # X的列数：IMAGE_HEIGHT * IMAGE_WIDTH 
        IMAGE_WIDTH  = np.shape(image)[1]
        
        MAX_CAPTCHA  = len(text)                   # Y的列数：MAX_CAPTCHA * CHAR_SET_LEN
        print("验证码文本字符串数：", MAX_CAPTCHA) 
        char_set = number
        CHAR_SET_LEN = len(char_set)
        
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
        dropout_percent = tf.placeholder(tf.float32)
        
        train_captcha_cnn()
    if train == 1:
        number = ['0','1','2','3','4','5','6','7','8','9']  
        IMAGE_HEIGHT = 60  
        IMAGE_WIDTH = 160  
        char_set = number
        CHAR_SET_LEN = len(char_set)

        text, image = gen_captcha_text_and_image()  
    
        f = plt.figure()  
        ax = f.add_subplot(111)  
        ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)  
        plt.imshow(image)  
       
        plt.show()  
        
        MAX_CAPTCHA = len(text)
        image = convert2gray(image)  
        image = image.flatten() / 255  
        
        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])  
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])  
        keep_prob = tf.placeholder(tf.float32) # dropout 
        
        predict_text = crack_captcha(image)  
        print("正确: {}  预测: {}".format(text, predict_text))  


# Mnist in Vec
# Model: y = x_T * w + b
# Train_Code
import tensorflow as tf

# 提取batch的函数
def read_and_decode(file_queue)  # 返回单个image和对应的label
  reader = tf.TFRecordReader()   # TFRecord是TF官方提供的一种数据格式，内部结构是PB(protocol buffer)
                                 # 可以吧许多零散的文件整合成为一个大的文件，可以提高效率，老师推荐把这个应用在实际的工程中
                                 # TFRecord读出来是一个string,可以通过tf.parse_single_example这样一个方法去解析string
                                 # 可参考：https://blog.csdn.net/happyhorizion/article/details/77894055 
                                 # TF还提供了：TextlenReader(读取文件的一行)；HoldfileReader(读整个文件)
  _, serialized_example = readed.read(file_queue)   # Python中'_'也可以作为一个变量，通常作为临时变量
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image_raw':tf.FixedLenFeature([], tf.string),
      'label':tf.FixedLenFeature([], tf.int64),
    })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([784])
  image = tf.cast(image, tf.float32) * (1. / 255)  # 归一化有助于加快收敛
  label = tf.cast(features['label'], tf.int32)
  return image. label

def read_image_batch(file_queue, batch_size)  # 从file_queue里读取batch_size个数据
  img, label = read_and_decode(file_queue)
  capacity = 3 * batch_size                   # 根据官方文档的推荐， 是考虑到后台的线程模型的一种比较好的大小
  image_batch, label_batch = tf.train.batch([img, label], batch_size = label_batch, capacity = capacity)
  one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))  # label的输入要把它转换成one-hot形式，相当于标签的向量化，0~9的分类用十个
                                                                   # 布尔型组成的向量构成
                                                                   # 原因在于：https://blog.csdn.net/m0_37561765/article/details/78207508
  return image_batch, one_hot_labels

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 这里OSS的过程：调用(request) -> ... -> TFRuntime -> XXXReaderOp(如这里的TFRecordReader) -> Env -> FileSystem(file://, oss://, hdfs://) 
train_file_path = "oss://....."   
train_image_filename_queue = tf.train.string_input_producer([train_file_path])  # 取出路径中的文件
train_images, train_labels = read_image_batch(train_image_filename_queue, 128)

x  = tf.reshape(train_images, [-1, 784])
y  = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.to_float(train_labels) 

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer     = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Prediction_Code
test_file_path = "oss://...."
test_image_filename_queue = tf.train.string_input_producer([test_file_path])
test_images, test_labels  = read_image_batch(test_image_filename_queue, 10000)

x_test = tf.reshape(test_images, [-1, 784])
y_pred = tf.nn.softmax(tf.matmul(x_test, w) + b)
y_test = tf.to_float(test_labels)
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 这里对y_pred = tf.nn.softmax(...)的理解：
# 1. 其本身没有参数去指定分类个数，都是由后面的 "tf.matmul(x_test, w) + b" => "[1, 784] * [784, 10] + [1, 10] == [1, 10]"
# 即做的10分类，所以在编写代码的时候是提前计算出来分类个数，而不是去直接指定。
# 2. 因为这是一个简单的例子，并没有做出更加复杂的诸如CNN或者线性回归的例子，所以也没有增加如激活函数等等的。

# Executive
init = tf.global_variables_initializer()

with tf.Session() as tf:
  sess.run(init)
  
  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess = sess, coord = coord)
  
  for i in range(101):
    sess.run(optimizer)  # 执行这一句是开始了模型的训练
    if i % 10 == 0:
      print("step: %d accuracy:%f" % (i, sess.run(accuracy)))
      
  coord.request_stop()
  coord.join(threads)
  # 71,72和79,80这几行代码在使用了tf_queue的时候就要用到，可以理解为一种固定格式
  print("done")

  

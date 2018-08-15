"""

@Author  : dilless
@Time    : 2018/8/11 21:31
@File    : cnn_mnist.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 下载并载入手写数据库(55000 张 * 28px * 28px)

# one_hot 是独热码
# 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 的十位数字
# 0: 1000000000
# 1: 0100000000
# 2: 0010000000
# ...
# 9: 0000000001
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# None 表示 Tensor 的第一个维度可以是任何长度
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.
output_y = tf.placeholder(tf.float32, [None, 10])  # 输出10个数字的标签
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])  # 改变形状之后的输入

# 从 Test 数据集选取 3000 个手写数字的图片和对应标签
test_x = mnist.test.images[:3000]
test_y = mnist.test.labels[:3000]

# 构建卷积神经网络
# 构建第一层卷积
conv1 = tf.layers.conv2d(
    inputs=input_x_images,  # 形状：[28 * 28 * 1]
    filters=32,  # 32 个过滤器，输出的深度是 32
    kernel_size=[5, 5],  # 过滤器在二维的大小是 [5 * 5]
    strides=1,  # 步长是 1
    padding='same',  # 让输出维持 [28 * 28]，需要在外围补 0 两圈
    activation=tf.nn.relu  # 激活函数为 Relu
)  # 形状会变成 [28, 28, 32]

# 第一层池化（亚采样）
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,  # 形状 [28, 28, 32]
    pool_size=[2, 2],  # 过滤器在二维的大小是 [2, 2]
    strides=2  # 步长是 2
)  # 形状是 [14, 14, 32] 因为步长是 2

# 构建第二层卷积
conv2 = tf.layers.conv2d(
    inputs=pool1,  # 形状：[14, 14, 32]
    filters=64,  # 64 个过滤器，输出的深度是 64
    kernel_size=[5, 5],  # 过滤器在二维的大小是 [5 * 5]
    strides=1,  # 步长是 1
    padding='same',  # same 表示输出维持 [28 * 28]，需要在外围补 0 两圈
    activation=tf.nn.relu  # 激活函数为 Relu
)  # 形状会变成 [14, 14, 64]

# 第二层池化
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,  # 形状 [14, 14, 64]
    pool_size=[2, 2],  # 过滤器在二维的大小是 [2, 2]
    strides=2  # 步长是 2
)  # 形状是 [7, 7, 64] 因为步长是 2

# 平坦化 (flat)
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # -1 表示根据其他参数推断该维度的大小

# 1024 个神经元的全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout: 丢弃 50%
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 经过 10 个神经元的全连接层，不用激活函数来做非线性化
logits = tf.layers.dense(inputs=dropout, units=10)  # 得到输出，形状是 [1, 1, 10]

# 计算误差 （Cross entropy(交叉熵)，再用 Softmax 计算百分比概率）
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# 用 Adam 优化器来最小化误差，学习率：0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)

# 计算精度，返回(accuracy, update_op)，会创建两个局部变量
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1), predictions=tf.argmax(logits, axis=1),)[1]

# 创建 Session
with tf.Session() as sess:
    # 初始化变量
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())  # 使用local_variables_initializer()因为要初始化精度生成的局部变量
    sess.run(init)

    for i in range(2000):
        batch = mnist.train.next_batch(50)  # 从 tarin 数据集里下一组 50 个样本
        train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
        if i % 100 == 0:
            test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
            print('Step=%d, Train loss=%.4f [Test accuracy=%.2f]' % (i, train_loss, test_accuracy))

    # 打印 20 个预测值和真实值对
    test_output = sess.run(logits, {input_x: test_x[:20]})
    inferenced_y = np.argmax(test_output, 1)
    writer = tf.summary.FileWriter('./log', sess.graph)
    print('Inferenced numbers: ', inferenced_y)  # 推测的数字
    print('Real numbers:       ', np.argmax(test_y[:20], 1))  # 真实的数字

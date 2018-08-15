"""

@Author  : dilless
@Time    : 2018/8/10 19:01
@File    : tensorboard.py
"""

import tensorflow as tf

# y = W * x + b
W = tf.Variable(2.0, dtype=tf.float32, name='Weight')  # 权重
b = tf.Variable(1.0, dtype=tf.float32, name='Bias')  # 偏差
x = tf.placeholder(dtype=tf.float32, name='Input')  # 输入
with tf.name_scope('Output'):
    y = W * x + b  # 输出

# 定义保存日志的路径
path = './log'

# 创建用于初始化所有变量的操作
init = tf.global_variables_initializer()

# 创建Session
with tf.Session() as sess:
    sess.run(init)  # 初始化变量
    writer = tf.summary.FileWriter(path, sess.graph)  # 写日志
    result = sess.run(y, {x: 3.0})  # 用dict填充x(placeholder)
    print('y = %s' % result)  # 打印 y = W * x + b 的值，为 7

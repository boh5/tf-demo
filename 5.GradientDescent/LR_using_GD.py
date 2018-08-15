"""
用梯度下降的优化方法来快速解决线性回归问题
@Author  : dilless
@Time    : 2018/8/10 19:58
@File    : LR_using_GD.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 构建数据
points_num = 100
vectors = []
# 用 Numpy 的正态随机分布函数生成 100 个点
# 用这些点的 (x, y) 坐标值对应线性方程 y = 0.1 * x + 0.2
# 权重(Weight) 0.1，偏差(Bias) 0.2
for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]  # x 坐标
y_data = [v[1] for v in vectors]  # y 坐标

# 图像 1：展示 100 个随机数据点
plt.plot(x_data, y_data, 'r*', label='Original data')  # 红色星形的点
plt.title('Linear Regression using Gradient Descent')
plt.legend()  # 展示 plot() 定义的label
plt.show()

# 构建线性回归模型
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 初始化权重
bias = tf.Variable(tf.zeros([1]))  # 初始化 Bias
y = weight * x_data + bias  # 模型计算出来的 y

# 定义 loss function (损失函数) 或 cost function (代价函数)
# 对 Tensor 的所有维度计算 ((y - y_data) ^ 2) / points_num
loss = tf.reduce_mean(tf.square(y - y_data))

# 用梯度下降的优化器来优化 loss function
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 设置学习率 0.5
train = optimizer.minimize(loss)

# 创建 Session
sess = tf.Session()

# 初始化数据流图中的所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练 20 步
for step in range(20):
    # 优化每弄一步
    sess.run(train)
    # 打印每一步的损失， 权重和偏差
    print('Step=%d, Loss=%f, [Weight=%f Bias=%f]' % (step, sess.run(loss), sess.run(weight), sess.run(bias)))

# 图像 2： 绘制所有的点并且绘制出最佳拟合的直线
plt.plot(x_data, y_data, 'r*', label='Original data')  # 红色星形的点
plt.title('Linear Regression using Gradient Descent')
plt.plot(x_data, sess.run(weight) * x_data + sess.run(bias), label='Fitted line')  # 拟合的线
plt.legend()  # 展示 plot() 定义的label
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 关闭 Session
sess.close()

"""

@Author  : dilless
@Time    : 2018/8/10 18:25
@File    : graph_session.py
"""

import tensorflow as tf

# 创建两个常量tensor
const1 = tf.constant([[2, 2]])
const2 = tf.constant([[4],
                      [4]])

multiple = tf.matmul(const1, const2)

# 尝试用print输出multiple的值
print(multiple)

sess = tf.Session()

# 用Session的run方法来实际运行multiple这个矩阵乘法操作
result = sess.run(multiple)

print(result)

if const1.graph is tf.get_default_graph():
    print('const1所在的图是当前上下文默认的图')

sess.close()

# 使用with来打开Session
with tf.Session() as sess:
    print(sess.run(multiple))

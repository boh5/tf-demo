"""

@Author  : dilless
@Time    : 2018/8/9 20:34
@File    : helloworld.py
"""

import tensorflow as tf

hw = tf.constant('Hello World!')

sess = tf.Session()

print(sess.run(hw))

sess.close()

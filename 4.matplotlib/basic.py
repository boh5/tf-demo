"""

@Author  : dilless
@Time    : 2018/8/10 19:34
@File    : basic.py
"""

# 引入 Matplotlib 的分模块 PyPlot
import matplotlib.pyplot as plt

import numpy as np

# 创建数据
x = np.linspace(-2, 2, 100)
# y = 3 * x + 4
y1 = 3 * x + 4
y2 = x ** 2

# 创建图像
# plt.plot(x, y)
plt.plot(x, y1)
plt.plot(x, y2)

# 显示图像
plt.show()

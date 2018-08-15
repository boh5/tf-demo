"""

@Author  : dilless
@Time    : 2018/8/10 19:39
@File    : figure.py
"""
# 引入 Matplotlib 的分模块 PyPlot
import matplotlib.pyplot as plt

import numpy as np

# 创建数据
x = np.linspace(-4, 4, 50)
# y = 3 * x + 4
y1 = 3 * x + 2
y2 = x ** 2

# 构建第一张图
plt.figure(num=1, figsize=(7, 6))
plt.plot(x, y1)
plt.plot(x, y2, color='red', linewidth=3.0, linestyle='--')
plt.show()

# 构建第二张图
plt.figure(num=2)
plt.plot(x, y2, color='green')

# 显示图像
plt.show()

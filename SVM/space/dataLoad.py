import numpy as np


def load_data():
    # 加载数据集 (506, 14)
    datafile = '../data.txt'
    data = np.loadtxt(datafile, dtype=float)
    # 将原始数据Reshape
    data = data.reshape([-1, 3])
    x = data[:, 0:-1]
    # 不使用 -1: 则结果为列向量
    y = data[:, -1]
    y[y == 0] = -1

    return x, y











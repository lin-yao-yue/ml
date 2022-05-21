import numpy as np


def load_data():
    # 加载数据集 (506, 14)
    datafile = '../perceptron_data.txt'
    data = np.loadtxt(datafile, dtype=float)
    # 将原始数据Reshape
    data = data.reshape([-1, 3])
    # 值替换
    # 将θ吸收进w
    data[data == 0] = -1

    x = data[:, 0:-1]
    y = data[:, -1:]
    theta = np.ones([len(data), 1])
    # 不增加维度的拼接是 cat
    x = np.concatenate((x, theta), axis=-1)

    """ 
    # 归一化
    # 返回每一列（一次计算涉及到的值必须都是 axis=0 上的值）的最大最小值，均值
    maximums = data.max(axis=0)
    minimums = data.min(axis=0)
    avgs = data.sum(axis=0) / data.shape[0]
    # 对训练集的每一列的数据进行归一化处理
    for i in range(4):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    """

    return x, y











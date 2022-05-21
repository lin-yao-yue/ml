import numpy as np


def load_data():
    # 加载数据集 (506, 14)
    datafile = '../housing_data.txt'
    data = np.fromfile(datafile, dtype=float, sep=' ')
    # 将原始数据Reshape
    data = data.reshape([-1, 14])
    # 归一化
    # 返回每一列（一次计算涉及到的值必须都是 axis=0 上的值）的最大最小值，均值
    maximums = data.max(axis=0)
    minimums = data.min(axis=0)
    avgs = data.sum(axis=0) / data.shape[0]
    # 对训练集的每一列的数据进行归一化处理
    for i in range(14):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 数据集拆分
    train_data = data[0:450, :]  # (450, 14)
    test_data = data[450:, :]  # (55, 14)

    return train_data, test_data











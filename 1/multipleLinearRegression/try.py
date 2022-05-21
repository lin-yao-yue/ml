import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # 导入房价数据
    datafile = '../housing_data.txt'
    # 空格作为分隔符，指定读取的数据类型，读取数据存储到数组中
    data = np.fromfile(datafile, dtype=float, sep=' ')
    # 将原始数据Reshape 并且拆分成训练集和测试集
    data = data.reshape([-1, 14])
    offset = int(data.shape[0]*0.8)
    # 归一化处理
    # 返回每一列（一次计算涉及到的值必须都是 axis=0 上的值）的最大最小值，均值
    maximums = data.max(axis=0)
    minimums = data.min(axis=0)
    avgs = data.sum(axis=0) / data.shape[0]
    # 对每一列的数据进行归一化处理
    for i in range(14):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    # 根据比例进行划分
    train_data = data[:offset]
    test_data = data[offset:]
    return train_data, test_data


class Network(object):
    def __init__(self, num_of_weight):
        # 随机产生w初始值
        # 使用相同的 seed 生成的随机数相同
        np.random.seed(0)
        '''
        rand: [0, 1) 的均匀分布
        randn: 标准正态分布，均值为0，方差为1
        normal: 指定均值和方差的正态分布
        randint(low,high,size=(2,2)) [low,high)下指定维度的随机整数
        '''
        self.w = np.random.randn(num_of_weight, 1)
        self.b = 0.

    def forword(self, x):   # 前向计算
        # 矩阵乘法，numpy 的广播机制
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):   # loss计算
        error = z - y
        cost = error * error
        # 求均值
        cost = np.mean(cost)
        return cost

    def gradient(self, x, y):   # 计算梯度
        z = self.forword(x)
        # 广播机制
        gradient_w = np.mean((z - y)*x, axis=0)
        # np.newaxis：在所放置的位置上增加一个维度
        gradient_w = gradient_w[:, np.newaxis]  # [13, ] -> [13, 1]
        gradient_b = np.mean((z - y), axis=0)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):     # 更新参数
        self.w = self.w - eta*gradient_w
        self.b = self.b - eta*gradient_b

    def train(self, x, y, iterations=100, eta=0.01):    # 训练代码
        losses = []
        for i in range(iterations):     # 训练多少轮
            z = self.forword(x)     # 前向计算
            loss = self.loss(z, y)  # 得到loss
            gradient_w, gradient_b = self.gradient(x, y)    # 计算梯度
            self.update(gradient_w, gradient_b, eta)    # 更新参数
            losses.append(loss)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, loss))
        return losses


train_data, test_data = load_data()
# [:-1] ：-1表示最后一项的下标，由于是右开区间，所以最后一项取不到
x = train_data[:, :-1]
y = train_data[:, -1:]
net = Network(13)
# 开始训练
losses = net.train(x, y, iterations=1000, eta=0.01)
# 画出损失函数变化趋势
plot_x = np.arange(1000)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()


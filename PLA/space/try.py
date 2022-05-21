#   -*- encoding:utf-8 -*-
#   The entire code is used to implement the perceptron
"""
Created on Tuesday, 24 December 14:58 2019
@author:Jeaten
@email:ljt_IT@163.com
"""
import numpy as np
import random

from space.dataLoad import load_data


class perceptron:
    def __init__(self):
        '''
         :param w:感知机的权重
         :param b:感知机的偏置
         :param learning_rate:学习率
        '''

        self.w = np.array([0, 0])
        self.b = 0
        self.learning_rate = 0.01

    def update(self, w, x, y, b):
        '''
        该函数用于参数的更新
        :param w: 权重
        :param x: 数据的特征
        :param y: 数据的标签
        :param b: 数据的偏置
        :return: 无
        '''
        self.w = w + self.learning_rate * x * y
        self.b = b + self.learning_rate * y

    def sign(self, w, x, b):
        '''
        该部分为符号函数
        :return 返回计算后的符号函数的值
        '''

        # 对于一维向量，计算的是内积
        return np.sign(np.dot(w, x) + b)

    def train(self, feature, label):
        '''
        该函数用于训练感知机
        :param feature: 特征
        :param label: 标签（数据点所属类别）
        :return: 返回最终训练好模型（参数）
        '''
        stop = True
        while stop:
            count = len(feature)
            for i in range(len(feature)):
                if self.sign(self.w, feature[i], self.b) * label[i] <= 0:
                    # print("分类错误！误分类点为:", feature[i])
                    self.update(self.w, feature[i], label[i], self.b)
                    print("w1: %f, w2: %f, θ: %f" % (self.w[0], self.w[1], self.b))
                else:
                    count -= 1
                if count == 0:
                    stop = False

        print("最终权重 w:", self.w, "最终偏置 b:", self.b)
        return self.w, self.b

    def train_rand(self, feature, label):
        '''
        该函数使用随机选择数据点来进行训练（随机梯度下降法）
        :param feature: 特征
        :param label: 标签（数据点所属类别）
        :return: 返回最终训练好模型（参数）
        '''
        stop = True
        while stop:
            count = len(feature)
            index = [i for i in range(len(feature))]
            random.shuffle(index)
            for i in index:
                if self.sign(self.w, feature[i], self.b) * label[i] <= 0:
                    print("分类错误！误分类点为:", feature[i])
                    self.update(self.w, feature[i], label[i], self.b)
                else:
                    count -= 1
            if count == 0:
                stop = False
        print("最终w:", self.w, "最终b:", self.b)
        return self.w, self.b


class show:
    def draw_curve(self, data, w, b):
        '''
        该函数应用于查看拟合效果
        :param data: 数据点
        :param w: 权重
        :param b: 偏置
        :return: 无
        '''
        import matplotlib.pyplot as plt
        coordinate_x = []
        for i in data:
            coordinate_x.append(i[0])
        coordinate_y = []
        for i in range(len(data)):
            coordinate_x.append(data[i][0])
            coordinate_y.append(data[i][1])
            if label[i] == 1:
                plt.plot(data[i][0], data[i][1], 'o' + 'r', label='right', ms=10)
            else:
                plt.plot(data[i][0], data[i][1], '*' + 'g', label='error', ms=10)
        d_y = []
        for i in coordinate_x:
            d_y.append(-(w[0] * i + b) / w[1])  # 一定的学习率下可能会有除0错误
        plt.plot(coordinate_x, d_y, )
        plt.show()
        return True


if __name__ == '__main__':
    data = load_data()
    feature = data[:, 0:-1]
    label = data[:, -1:]

    x = np.array(feature)
    y = np.array(label)
    neuron = perceptron()
    show = show()
    w, b = neuron.train(x, y)  # 按序调整
    # w,b=neuron.train_rand(x,y)#使用随机梯度下降法
    show.draw_curve(feature, w, b)

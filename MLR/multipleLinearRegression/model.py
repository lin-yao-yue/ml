import numpy as np
import matplotlib.pyplot as plt
from multipleLinearRegression.dataLoad import load_data


class Model(object):
    def __init__(self, num_inputs):
        self.w = np.random.randn(num_inputs, 1)
        self.b = 0

    def foward(self, x):
        return np.dot(x, self.w) + self.b

    def loss(self, y, y_hat):
        v = (y - y_hat) ** 2 / 2
        return np.mean(v)

    def gradient(self, x, y, y_hat):
        # 多元线性回归的梯度计算
        gradient_w = np.mean((y_hat - y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]  # [13,] -> [13, 1]
        gradient_b = np.mean(y_hat - y)
        return gradient_w, gradient_b

    def train(self, x, y, epochs, lr):
        ans = []
        for epoch in range(epochs):
            y_hat = self.foward(x)
            los = self.loss(y, y_hat)
            ans.append(los)
            print('epoch %d, loss %f' % (epoch + 1, los))
            # 更新参数
            gradient_w, gradient_b = self.gradient(x, y, y_hat)
            self.w = self.w - lr * gradient_w
            self.b = self.b - lr * gradient_b
        return ans




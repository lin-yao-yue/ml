import numpy as np


class Model:
    def __init__(self):
        self.w = np.random.randn(3, 1)
        self.lr = 0.1

    def sign(self, x):
        x[x >= 0] = 1
        x[x < 0] = -1
        # batch*1
        return x

    def update(self, x, y):
        delta_w = self.lr * np.matmul(x.T, y - self.forward(x))
        self.w += delta_w

    def forward(self, x):
        # batch*1
        return self.sign(np.matmul(x, self.w))

    def train(self, x, y):
        cnt = 0
        for i in range(100):
            cnt += 1
            self.update(x, y)
            print("train: %d, w1: %f, w2: %f, θ: %f" % (cnt, self.w[0][0], self.w[1][0], self.w[2][0]))




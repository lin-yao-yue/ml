import numpy as np


class Model(object):
    def __init__(self, num_inputs):
        self.w = np.random.randn(num_inputs, 1)

    def forward(self, x):
        return np.matmul(x, self.w)

    def optimize(self, x0, x1, u0, u1):

        s_w = np.zeros((x0.shape[1], x0.shape[1]))
        s_w += np.matmul((x0-u0).T, (x0-u0))
        s_w += np.matmul((x1-u1).T, (x1-u1))
        # 使用奇异值分解
        u, s, v = np.linalg.svd(s_w)
        # s 是对角线元素的一维向量，使用 0 填充成对角阵
        s_w_inv = np.matmul(np.matmul(np.transpose(v), np.linalg.inv(np.diag(s))), np.transpose(u))
        self.w = np.matmul(s_w_inv, (u0 - u1).T)

        '''
        # 行向量转换成列向量 (4, 1)
        u0_t = np.transpose(u0)
        u1_t = np.transpose(u1)
        s_w = np.zeros((u0_t.shape[0], u0_t.shape[0]))
        for i in range(x0.shape[0]):
            x0_t = np.transpose(x0[i])
            s_w = s_w + np.matmul((x0_t-u0_t), np.transpose(x0_t-u0_t))
        for i in range(x1.shape[0]):
            x1_t = np.transpose(x1[i])
            s_w = s_w + np.matmul((x1_t-u1_t), np.transpose(x1_t-u1_t))
        
        # 直接求逆矩阵，不使用奇异值分解
        # self.w = np.matmul(np.linalg.inv(s_w), (u0_t - u1_t))
        # 使用奇异值分解
        u, s, v = np.linalg.svd(s_w)
        # s 是对角线元素的一维向量，使用 0 填充成对角阵
        s_w_inv = np.matmul(np.matmul(np.transpose(v), np.linalg.inv(np.diag(s))), np.transpose(u))
        self.w = np.matmul(s_w_inv, (u0 - u1).T)
        '''




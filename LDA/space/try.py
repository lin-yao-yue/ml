import matplotlib.pyplot as plt
import numpy as np
from space.dataLoad import load_data


def draw1(data, theta, num, colors):
    x = [[] for i in range(2)]
    for ele in data:
        x[ele[-1]].append(ele[:-1])
    # 绘制原始数据
    for i in range(num):
        x[i] = np.array(x[i])
        plt.scatter(x[i][:, 0], x[i][:, 1], color=colors[i])
    # 绘制被的映射直线
    plt.plot([0, theta[0] * 15], [0, theta[1] * 15])
    # 绘制映射到直线上的点
    for i in range(num):
        for ele in x[i]:
            ta = theta * np.dot(ele, theta)
            plt.plot([ele[0], ta[0]], [ele[1], ta[1]], color=colors[i], linestyle="--")
            plt.scatter(ta[0], ta[1], color=colors[i])
    plt.show()


def c2(data, num):
    # n = 4
    n = data.shape[1] - 1
    # 存储x值 x = [[]]
    x = [[] for i in range(num)]
    # 存储u值 u = [[]]
    u = [[] for i in range(num)]
    # 存储Sw
    sw = np.zeros([n, n])
    for ele in data:
        # ele[-1]: 1或0，根据 y 值分类填充
        # x.shape = (2, size, 4)
        x[ele[-1]].append(ele[:-1])
    for i in range(num):
        x[i] = np.array(x[i])
        # u.shape = (2, 4)
        u[i] = np.mean(x[i], axis=0)
    print("1计算平均值：\n", u)
    for i in range(num):
        x[i] = x[i] - u[i]
        sw = sw + np.dot(x[i].T, x[i])
    print("2x_i去中心化:\n", x)
    print("3计算散度矩阵S_w:\n", sw)
    # 计算theta
    theta = np.dot(np.linalg.inv(sw), (u[0] - u[1]).T)
    '''
    # 单位化
    fm = 0
    for i in range(n):
        fm = fm + theta[i] ** 2
    return theta / np.sqrt(fm)
    '''
    return theta


train_data, test_data = load_data()
data = train_data
colors = ['red', 'green']
theta = c2(data, 2)
'''
print(theta)
[ 1.55923510e-04 -1.52587891e-05 -7.15255737e-07  3.80386664e-05]
'''
train_x = train_data[:, 0:-1]
train_y = train_data[:, -1:]
train_x_0 = []
train_x_1 = []
for i in range(train_x.shape[0]):
    if train_y[i][0] == 0:
        train_x_0.append(train_x[i])
    else:
        train_x_1.append(train_x[i])

train_x_0 = np.array(train_x_0)
train_x_1 = np.array(train_x_1)

pre_0 = np.matmul(train_x_0, theta)
pre_1 = np.matmul(train_x_1, theta)

# 散点图
plt.scatter(pre_0, pre_0, color="red", marker='o', label='y=0')
plt.scatter(pre_1, pre_1, color="blue", marker='+', label='y=1')
plt.legend(loc='lower right')
plt.show()

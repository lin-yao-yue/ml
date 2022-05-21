from space.dataLoad import load_data
import numpy as np
from space.model import Model
import matplotlib.pyplot as plt
import random

# (450, 14)  (55, 14)
train_data, test_data = load_data()

train_x = train_data[:, 0:-1]
train_y = train_data[:, -1:]
test_x = test_data[:, 0:-1]
test_y = test_data[:, -1:]

train_x_0 = []
train_x_1 = []
for i in range(train_x.shape[0]):
    if train_y[i][0] == 0:
        train_x_0.append(train_x[i])
    else:
        train_x_1.append(train_x[i])

train_x_0 = np.array(train_x_0)
train_x_1 = np.array(train_x_1)

u0 = train_x_0.mean(axis=0)
# print(u0.shape) (4,)
u0 = u0.reshape(1, -1)
u1 = train_x_1.mean(axis=0)
u1 = u1.reshape(1, -1)

model = Model(train_x.shape[1])

model.optimize(train_x_0, train_x_1, u0, u1)


# 散点图

# 画布大小
plt.figure(figsize=(16, 4))
# 画布中的子图
plt.subplot(1, 2, 1)
pre_0 = model.forward(train_x_0)
pre_1 = model.forward(train_x_1)
pre = model.forward(test_x)
plt.scatter(pre_0, pre_0, color="red", marker='o', label='train_y=0')
plt.scatter(pre_1, pre_1, color="blue", marker='+', label='train_y=1')
plt.scatter(pre, pre, color="green", marker='.', s=5, label='test')
plt.legend(loc='lower right')

# 根据投影结果赋值预测
pre_y = []
for i in range(pre.shape[0]):
    if pre[i] <= -0.0020:
        pre_y.append(1)
    elif pre[i] >= 0.0019:
        pre_y.append(0)
    else:
        # 随机赋值
        pre_y.append(random.randint(0, 1))

# 计算准确率
# np.array(pre_y).shape : (148, ) 要 reshape 成 (148, 1)
# (148, ) 既可表示行向量，也可表示列向量，明确指定时要 reshape
pre_y = np.array(pre_y).reshape(-1, 1)
acc = (pre_y.shape[0] - np.sum((pre_y-test_y)**2))/pre_y.shape[0]
# 计算过程要放在()里，否则表示重复输出100次
print('accuracy: %f' % (acc*100)+'%')

# 预测结果画图
pre_y_0 = []
pre_y_1 = []
for i in range(pre_y.shape[0]):
    if pre_y[i] == 0:
        pre_y_0.append(pre[i])
    else:
        pre_y_1.append(pre[i])

plt.subplot(1, 2, 2)
pre_y_0 = np.array(pre_y_0)
pre_y_1 = np.array(pre_y_1)
# 设置坐标轴范围
plt.xlim(-0.01, 0.01)
plt.ylim(-0.01, 0.01)

plt.scatter(pre_y_0, pre_y_0, color="red", marker='o', label='test_y=0')
plt.scatter(pre_y_1, pre_y_1, color="blue", marker='+', label='test_y=1')
plt.legend(loc='lower right')
plt.show()


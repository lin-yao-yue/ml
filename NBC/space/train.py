import numpy as np
from space.dataLoad import load_data_train
from space.dataLoad import load_data_test

# (1934, 1025)
data = load_data_train()
# 列表初始化
p_c = [0]*10

label = data[:, -1]
# print(label.shape) (1934,)
for i in range(len(label)):
    # char 转 int
    p_c[int(label[i])] += 1

p_c = np.array(p_c, dtype=float)
# print(p_c) [189 198 195 199 186 187 195 201 180 204]

# 某类样本的某个位置取值为0/1的 样本数量
p_x_0 = []
p_x_1 = []

for i in range(10):
    # int 转 char 会乱码，转 str
    cur_label_data = data[data[:, -1] == str(i)]
    # 去掉最后的label列
    cur_label_data = cur_label_data[:, 0:-1]
    x_i_0 = []
    x_i_1 = []
    for j in range(cur_label_data.shape[1]):
        x_0 = cur_label_data[cur_label_data[:, j] == str(0)]
        x_1 = cur_label_data[cur_label_data[:, j] == str(1)]
        x_i_0.append(len(x_0))
        x_i_1.append(len(x_1))
    p_x_0.append(x_i_0)
    p_x_1.append(x_i_1)

p_x_0 = np.array(p_x_0, dtype=float)
p_x_1 = np.array(p_x_1, dtype=float)
# print(p_x_0.shape) (10, 1024)
# print(p_x_1.shape) (10, 1024)

# 计算p(c), p(xi|c)
for i in range(10):
    p_x_0[i] = p_x_0[i] / p_c[i]
    p_x_1[i] = p_x_1[i] / p_c[i]

p_c = p_c/len(data)
'''
print(np.sum(p_x_0, axis=1))
print(np.sum(p_x_1, axis=1))
print(p_c)
'''
'''--------------------------------test------------------------------------'''
test = load_data_test()
# print(test.shape) (946, 1025)

res = []
for i in range(len(test)):
    label = test[i, -1]
    cur_test = test[i, 0:-1]
    # 计算0~9的似然
    mul_max = -1
    mul_max_index = -1
    
    for j in range(10):
        # 元素乘积
        mul_0 = np.prod(p_x_0[j][[cur_test == str(0)]])
        mul_1 = np.prod(p_x_1[j][[cur_test == str(1)]])
        mul_res = p_c[j] * mul_0 * mul_1
        if mul_res > mul_max:
            mul_max = mul_res
            mul_max_index = j
    res.append(str(mul_max_index))

res = np.array(res)
print(res)
print('accuracy: %lf' % (len(res[res == test[:, -1]])/len(test)))


import queue

import numpy as np

from space.buildTree import create_tree
from space.dataLoad import load_data
from space.model import calculate_gain, IVa
from space.node import Node
from space.drawTree import *

x_y = load_data()

values = [["young", "pre-presbyopic", "presbyopic"],
          ["myope", "hypermetrope"],
          ["no", "yes"],
          ["reduced", "normal"],
          ["hard", "soft", "not lenses"]]

# 二维列表的构造方法
data = [["" for col in range(x_y.shape[1])] for row in range(x_y.shape[0])]

for j in range(x_y.shape[1]):
    for i in range(x_y.shape[0]):
        index = x_y[i][j]
        data[i][j] = values[j][index - 1]

# 数值矩阵转换为字符串矩阵
data = np.array(data)

# 属性取值矩阵
attrs = []
for i in range((data.shape[1] - 1)):
    values = [row[i] for row in data]
    uniqueValues = set(values)
    attrs.append(list(uniqueValues))

attrs_name = ["age", "prescrip", "astigmatic", "tearRate"]

# 根节点
root = Node(key=-1, pre_choice="")
create_tree(data, attrs, attrs_name, root)

# 打印信息增益率
gain, gain_mean = calculate_gain(data=data, detail_features=attrs)
gain_ratio = []
for i in range(len(gain)):
    gain_ratio.append(gain[i]/IVa(data=data, attr_index=i))

print(gain_ratio)
# [0.024856426337024264, 0.039510835423565815, 0.3770052300114771, 0.5487949406953985]


# 既可以返回字典，也可以返回字符串
def plot_tree(root):
    cur_tree = {}
    sub_tree = {}
    if len(root.childs) != 0:
        for node in root.childs:
            sub_tree[node.pre_choice] = plot_tree(node)
        cur_tree[root.key] = sub_tree
        # 返回字典
        return cur_tree
    else:
        # 返回字符串
        return root.key


tree = plot_tree(root)
print(tree)
createPlot(tree)

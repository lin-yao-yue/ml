import numpy as np

from space.node import Node
from space.model import entropy, calculate_gain, IVa


# 判断样本是否在所有属性上都只有一个取值，以致无法划分
# attribute：属性
def same_data(data, attrs):
    for i in range(len(attrs)):
        if len(set(data[:, i])) > 1:
            return False
    return True


# attrs:属性的具体形式
def create_tree(data, attrs, attrs_name, root):
    # 3个退出条件
    # 1，如果数据为空，不能划分，此时这个叶节点不知标记为哪个分类了
    if len(data) == 0:
        return

    # 2，如果属性集为空，或所有样本在所有属性的取值相同，无法划分，返回样本最多的类别
    if len(attrs) == 0 or same_data(data, attrs):
        class_set = list(set(data[:, -1]))
        max_len = 0
        index = 0
        # 遍历所有属性，计算每一种属性所占的数量，找出最多的 attr
        for i in range(len(class_set)):
            if len(data[data[:, -1] == class_set[i]]) > max_len:
                max_len = len(data[data[:, -1] == class_set[i]])
                index = i

        # 属性集为空 or 样本在所有属性取值相同：不知道根据哪个属性进行划分，所以不用记录属性值
        # 标记 label
        root.key = class_set[index]
        return

    # 3，如果当前节点包含同一类的样本，无需划分
    if len(set(data[:, -1])) == 1:
        root.key = data[0, -1]
        return

    # 从 data 中选出最优划分属性

    # 不同属性的信息增益
    gain_result, mean = calculate_gain(data, attrs)
    max = 0
    max_index = -1
    # 求增益率最大
    for i in range(len(gain_result)):
        if gain_result[i] >= mean:
            # 固有值
            iva = IVa(data, i)
            # 增益率
            if gain_result[i] / iva > max:
                max = gain_result[i] / iva
                max_index = i

    # 确定当前 node 的 key
    root.key = attrs_name[max_index]

    # 增益率最大的属性的每一种取值进行划分
    for j in range(len(attrs[max_index])):
        part_data = data[data[:, max_index] == attrs[max_index][j]]
        # 删除数据中该属性所在的列
        part_data = np.delete(part_data, max_index, axis=1)
        # 添加节点,记录所划分属性的取值
        root.add_node(Node(key="", pre_choice=attrs[max_index][j]))
        # 删除该属性
        new_attrs = attrs[0:max_index]
        new_attrs_name = attrs_name[0:max_index]
        # 一维数组的拼接
        new_attrs.extend(attrs[max_index + 1:])
        new_attrs_name.extend(attrs_name[max_index + 1:])
        # 递归构造子树
        create_tree(part_data, new_attrs, new_attrs_name, root.childs[j])


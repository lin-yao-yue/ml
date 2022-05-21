import numpy as np


# 样本 的 信息熵---样本的混乱程度
# data:like np.array，某个属性的某个取值下的样本
# data.shape=(num_data,data_features+1) 即属性与label放一起了
def entropy(data):
    class_set = list(set(data[:, -1]))
    result = 0
    length = len(data)

    for i in range(len(class_set)):
        # data[:, -1] == class_set[i] 得到一个np.array,符合条件值为true，否则为false
        # 使用该 bool 值 array 对 data 进行切片
        l = len(data[data[:, -1] == class_set[i]])
        p = l / length
        # 防止某类未出现，概率为0
        if p > 0:
            result -= p * np.log2(p)
    return result


# 计算不同属性的信息增益
# detail_features：属性构成的list，每个属性的取值构成list元素，即也是list (特征数量，每个特征的取值)
# 每个属性的可取值数量可能不同，所以用 list 存储
def calculate_gain(data, detail_features):
    '''返回各属性对应的信息增益及平均值'''
    result = []
    # 总样本的信息熵
    ent_data = entropy(data)
    # 计算每个属性的信息增益
    for i in range(len(detail_features)):
        res = ent_data
        for j in range(len(detail_features[i])):
            # 根据指定属性 d 的取值 a 对数据集进行划分
            part_data = data[data[:, i] == detail_features[i][j]]
            length = len(part_data)
            #                           划分样本的信息熵
            res -= length / len(data) * entropy(part_data)
        result.append(res)
    return result, np.array(result).mean()


# 计算某个属性的固有值
# attr_index 某属性在矩阵中的列号
def IVa(data, attr_index):
    attr_values = list(set(data[:, attr_index]))
    v = len(attr_values)
    res = 0
    for i in range(v):
        # 根据属性的取值对 data 进行提取
        part_data = data[data[:, attr_index] == attr_values[i]]
        p = len(part_data) / len(data)
        res -= p * np.log2(p)
    return res


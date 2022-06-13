import numpy as np
import os


def load_data_train():
    path = '../traindata'
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    data = []
    for file in files:
        label = file[0]
        location = os.path.join(path, file)
        cur_data = np.loadtxt(location, dtype=np.str_)
        cur_data_str = ''
        for i in range(len(cur_data)):
            cur_data_str += cur_data[i]
        cur_data_str += label
        cur_data_char_lis = []
        for i in range(len(cur_data_str)):
            cur_data_char_lis.append(cur_data_str[i])
        data.append(cur_data_char_lis)
    data = np.array(data)
    # print(data.shape) (1934, 1025)
    return data


def load_data_test():
    path = '../testdata'
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    data = []
    for file in files:
        label = file[0]
        location = os.path.join(path, file)
        cur_data = np.loadtxt(location, dtype=np.str_)
        cur_data_str = ''
        for i in range(len(cur_data)):
            cur_data_str += cur_data[i]
        cur_data_str += label
        cur_data_char_lis = []
        for i in range(len(cur_data_str)):
            cur_data_char_lis.append(cur_data_str[i])
        data.append(cur_data_char_lis)
    data = np.array(data)
    return data

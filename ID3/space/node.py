class Node:
    def __init__(self, key, pre_choice):
        self.childs = []
        # 对当前节点的 key 属性进行划分 或 当前节点的 label 值
        self.key = key
        # 当前节点是根据某属性的 value 划分得到
        self.pre_choice = pre_choice

    def add_node(self, node):
        self.childs.append(node)



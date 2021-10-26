
import pickle
import numpy as np
import matplotlib.pyplot as plt



'''递归获取树的深度(深度优先遍历)'''
def getTreeDepth(DTree):
    max_depth = 0
    root = list(DTree.keys())[0]
    son_node = DTree[root]
    # 返回当前层的最大深度
    for key in son_node.keys():
        if type(son_node[key]).__name__ == 'dict':
            # 递归计算子节点的深度
            this_depth = getTreeDepth(son_node[key])
        else: 
            # 如果子节点是叶子结点，则子节点的深度为1
            this_depth = 1
        # 子树的深度取决于最大的子节点的深度
        if this_depth > max_depth:
            max_depth = this_depth
    # 树的深度 = 1(根节点) + 子节点的深度
    return max_depth + 1



'''递归获取树的叶子结点类别(深度优先遍历)'''
def getLeafCls(DTree, img_index):
    root = list(DTree.keys())[0]
    son_node = DTree[root]
    for key in son_node.keys():
        if type(son_node[key]).__name__ == 'dict':
            depth = getTreeDepth(son_node[key])
            img_index[root] += depth  
            getLeafCls(son_node[key], img_index)
        else: 
            return
    return img_index



Tree = pickle.load(open('digitsTree(gain_ratio).txt', 'rb'))
img_idx = np.zeros(784)
img_idx = getLeafCls(Tree, img_idx).reshape(28,28)
plt.imshow(img_idx)
plt.show()
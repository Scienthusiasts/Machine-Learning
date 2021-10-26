import numpy as np     
import pickle
import copy
from collections import Counter

from myutils import readFile, createPlot


class decisionTree:

    def __init__(self, X_train, y_train, X_test, y_test, label, mode="gain"):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.label = label
        self.mode = mode


    '''计算给定数据集(标签)的总信息熵'''
    # y:数据的标签, size=[batches,]
    def calcShannonEnt(self, y):
        entropy = 0.  # 信息熵
        size = len(y) # 数据集大小
        classes_idx_num = dict(Counter(y)) # 统计每类标签下包含的数据个数
        # 计算信息熵
        for key in classes_idx_num.keys():
            # 计算第key个标签的信息熵分量
            prob = classes_idx_num[key] / size # 用出现频率表示概率
            entropy -= prob * np.log2(prob)    # 信息熵计算公式
        return entropy


    '''选取特定维度下特定值的数据，返回剔除这个维度后的结果
    (相当于一次决策过程生成一个子数据集)
    '''
    # X:     输入数据
    # y:     输入数据的标签
    # label: 输入数据不同维度的属性
    # dim：  当前是第几维度
    # val：  该维度的指定取值
    def datasetSplit(self, X, y, label, dim, val):
        sub_X = []  # 保存划分后的子数据集
        sub_y = []  # 保存子数据集的标签
        # 遍历所有数据
        for i in range(X.shape[0]):
            # 如果指定维度下的特征等于指定值
            if X[i, dim] == val:
                data = X[i, :]
                # 则选择这个数据并剔除这一维度
                sub_X.append(np.delete(data, dim, 0))
                sub_y.append(y[i])
        # 删除已决策维度的标签
        sub_label = np.delete(label, dim, 0)
        return np.array(sub_X), np.array(sub_y), sub_label


    '''划分选择:使用信息增益'''
    # X:     输入数据, size=(batches, features)
    # y:     类别标签, size=(batches,)
    # dim:   当前是第几维度
    # num_D: 数据总数
    # ent_D: 数据集整体熵
    def Gain(self, X, y, num_D, ent_D, dim):
        a = X[:, dim] # 获取数据第dim维度
        v = set(a)    # 获取数据第dim维度可能的取值
        ent_a = 0     # 数据第dim维度的信息熵
        # 计算数据第dim维度的信息增益:
        for i in v:
            # 第dim维度第i个取值出现的频率
            prob_a_v = np.sum(a==i)/num_D
            # 第dim维度第i个取值下的信息熵
            ent_a_v_cls = self.calcShannonEnt(y[np.where(a==i)])
            ent_a += prob_a_v * ent_a_v_cls
        return ent_D - ent_a


    '''划分选择:使用信息增益率'''
    # X:     输入数据, size=(batches, features)
    # y:     类别标签, size=(batches,)
    # dim:   当前是第几维度
    # num_D: 数据总数
    # ent_D: 数据集整体熵
    def gainRatio(self, X, y, num_D, ent_D, dim):
        a = X[:, dim] # 获取数据第dim维度
        # 首先求取数据的信息增益
        gain = self.Gain(X, y, num_D, ent_D, dim)
        # 求取数据的固有值
        IV = self.calcShannonEnt(a)
        # 信息增益率=固有值*信息增益
        if IV == 0 :
            return 0 
        else :
            return (gain / IV)








    # 计算某一维度相对于标签的基尼指数
    def Gini(self, y):
        size = len(y) # 数据集大小
        gini_total = 0
        classes_idx_num = dict(Counter(y)) # 统计每类标签下包含的数据个数
        # 计算基尼系数:
        for key in classes_idx_num.keys():
            # 计算第key个标签的基尼系数分量
            prob = classes_idx_num[key] / size # 用出现频率表示概率
            gini_total += prob * prob
        return 1 - gini_total


    '''划分选择:使用基尼系数'''
    # X:     输入数据, size=(batches, features)
    # y:     类别标签, size=(batches,)
    # dim:   当前是第几维度
    # num_D: 数据总数
    def GiniIdx(self, X, y, num_D, dim):
        a = X[:, dim] # 获取数据第dim维度
        v = set(a)    # 获取数据第dim维度可能的取值
        gini_a = 0
        # 计算数据第dim维度的信息增益:
        for i in v:
            # 第dim维度第i个取值出现的频率
            prob_a_v = np.sum(a==i)/num_D
            gini_a_v = self.Gini(y[np.where(a==i)])
            gini_a += prob_a_v * gini_a_v
        return gini_a






    '''选取最佳划分维度'''
    def bestFeature(self, X, y):
        # 数据总信息熵
        ent_D = self.calcShannonEnt(y)
        # 数据的大小
        num_D = len(y)
        # 最佳信息增益对应的维度(初始化-1)
        best_dim = -1

        # 遍历所有维度寻找最佳信息增益
        if self.mode == "gain":
            # 最佳信息增益(初始化0)
            best_gain = 0
            for i in range(X.shape[1]):
                gain = self.Gain(X, y, num_D, ent_D, i) # 信息增益
                # 更新
                if best_gain < gain:
                    best_gain = gain 
                    best_dim = i
        if self.mode == "gain_ratio":
            # 最佳信息增益(初始化0)
            best_gain = 0
            for i in range(X.shape[1]):
                gain = self.gainRatio(X, y, num_D, ent_D, i) # 信息增益率
                # 更新
                if best_gain < gain:
                    best_gain = gain 
                    best_dim = i
        if self.mode == "gini":
            # 最佳信息增益(初始化inf)
            best_gain = np.inf
            for i in range(X.shape[1]):
                gain = self.GiniIdx(X, y, num_D, i)           # 基尼系数
                # 更新
                if best_gain > gain:
                    best_gain = gain 
                    best_dim = i

        return best_dim


    '''递归生成决策树'''
    def createTree(self, X, y, label):
        # 类别完全相同则停止继续划分
        if len(set(y)) == 1:
            return y[0]
        # 遍历完所有的维度返回出现次数最多的类别
        if X.shape[1] == 1:
            return np.argmax(np.bincount(y))
        # 选取最佳划分维度    
        best_dim = self.bestFeature(X, y)
        # 以当前维度为根节点创建决策树(字典的形式)
        DTree = {label[best_dim]:{}}
        # 获取当前维度的所有可能取值:
        dim_val = X[:, best_dim]
        unique_vals = set(dim_val)
        # 递归的创建决策树(深度优先遍历)
        for val in unique_vals:
            sub_X, sub_y, sub_label = self.datasetSplit(X, y, label, best_dim, val)
            # print(sub_y.shape)
            DTree[label[best_dim]][val] = self.createTree(sub_X, sub_y, sub_label)
        return DTree


    '''递归生成决策树'''
    def createTree_with_cut(self, X, y, label):
        # print(X.shape)
        # 类别完全相同则停止继续划分
        if len(set(y)) == 1:
            return y[0]
        # 遍历完所有的维度返回出现次数最多的类别
        if X.shape[1] == 1:
            return np.argmax(np.bincount(y))
        # 选取最佳划分维度    
        best_dim = self.bestFeature(X, y)
        print(best_dim)
        # 以当前维度为根节点创建决策树(字典的形式)
        DTree = {label[best_dim]:{}}
        #########################################################
        # 计算精度(剪枝)
        max_cls = max(list(y), key=list(y).count)
        before_acc = sum(self.y_test==max_cls)/self.y_test.shape[0]
        #########################################################
        # 获取当前维度的所有可能取值:
        dim_val = X[:, best_dim]
        unique_vals = set(dim_val)
        # 递归的创建决策树(深度优先遍历)
        for val in unique_vals:
            sub_X, sub_y, sub_label = self.datasetSplit(X, y, label, best_dim, val)
            # print(sub_y.shape)
            DTree[label[best_dim]][val] = self.createTree_with_cut(sub_X, sub_y, sub_label)
        #########################################################
        # 剪枝
        after_acc = self.calcAccuracy(DTree, label)
        if(before_acc > after_acc):
            return max_cls
        #########################################################

        return DTree

    '''计算测试集精度'''
    def calcAccuracy(self, DTree, label):
        acc = 0
        for i in range(self.y_test.shape[0]):
            acc += self.predict(DTree, self.X_test[i,:], label) == self.y_test[i]
        return acc / self.y_test.shape[0]


    '''创建决策树(调用递归函数)'''
    def initDTree(self):
        DTree = self.createTree(self.X_train, self.y_train, self.label)
        # DTree = self.createTree_with_cut(self.X_train, self.y_train, self.label)
        # 决策树的结构是一个嵌套字典:
        self.DTree = DTree


    '''决策树搜索预测过程(DFS)'''
    '''缺点在于每次只能喂入一条数据'''
    def predict(self, DTree, X, label):
        root = list(DTree.keys())[0]
        son_node = DTree[root]
        # 寻找当前决策节点在数据的哪一维度:
        index_0 = np.where(label == root)[0]
        index_1 = np.where(X == root)[0]
        index = index_1[0] if index_0.size == 0 else index_0[0]
        # 深度优先遍历:
        class_label = -1
        for key in son_node.keys():
            if X[index] == key:
                if type(son_node[key]).__name__ == 'dict':
                    # 如果不是叶子节点就递归搜索
                    class_label = self.predict(son_node[key], X, label)
                else:
                    # 如果是叶子节点就返回对应的类别
                    class_label = son_node[key]
        # 若决策树没有数据当前维度下的取值对应的决策属性，则直接返回该节点下数量最多的类别
        return self.getLeafCls(DTree) if class_label == -1 else class_label



    '''递归获取树的叶子结点类别(深度优先遍历)'''
    def _getLeafCls(self, DTree, leaf):
        root = list(DTree.keys())[0]
        son_node = DTree[root]
        for key in son_node.keys():
            if type(son_node[key]).__name__ == 'dict':
                # 递归计算子节点
                leaf = self._getLeafCls(son_node[key], leaf)
            else: 
                leaf.append(son_node[key])
        return leaf

    '''获取树的叶子结点数量最多的类别'''
    def getLeafCls(self, DTree):
        leaf_cls = []
        leaf_cls = self._getLeafCls(DTree, leaf_cls)

        # 返回当前节点下数量最多的叶子结点的类别：
        best_cls = max(leaf_cls, key=leaf_cls.count)
        return best_cls




    '''保存决策树模型'''
    def saveTree(self, path):
        # 以二进制文件保存
        file = open(path, 'wb')
        pickle.dump(self.DTree, file)

    '''读取决策树模型'''
    def loadTree(self, path):
        # 以二进制文件读取
        file = open(path, 'rb')
        self.DTree = pickle.load(file)








if __name__ == '__main__':
    path = './lenses.txt'
    X, y, label = readFile(path)
    DT = decisionTree(X, y, label)
    DT.initDTree()

    # createPlot(DT.DTree)
    DT.saveTree('./dtree.txt')
    print(DT.DTree)
    DT.loadTree('./dtree.txt')
    print(DT.predict(DT.DTree, X[9,:], label))
    print(DT.getLeafCls(DT.DTree))



import numpy as np


class kNN():
    def __init__(self, k, X_train, y_train, X_test):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.neighbors = np.zeros((len(self.X_test), len(self.X_train)))

    # 欧氏距离
    def EuclDist(self, x0, x1):
        return np.sum(np.square(x1 - x0))

    # 计算当前数据与标签数据的距离
    def Allneighbors(self):
        for i in range(len(self.X_test)):
            for j in range(len(self.X_train)):
                self.neighbors[i, j] = self.EuclDist(self.X_test[i], self.X_train[j]) # 计算欧式距离

    # 下标转为类别(分类问题)
    def index2label(self, index):
        knearest = self.y_train[index][:self.X_test.shape[0]] # 获取下标对应的标签
        # 统计K近邻的大多数:
        predict = []
        for i in range(self.X_test.shape[0]):
            predict.append(np.argmax(np.bincount(knearest[i]))) # 统计出现次数最多的类别
        return np.array(predict)

    # 下标转为数值(回归问题)
    def index2value(self, index):
        knearest = self.y_train[index][:self.X_test.shape[0]] # 获取下标对应的标签
        # 统计K近邻的大多数:
        predict = np.mean(knearest, axis=1) # 预测结果为k近邻的均值
        return predict.reshape(-1)

    # kNN算法主干
    def kNN(self, mode="classification"):
        # 1.计算距离
        self.Allneighbors() 
        # 2.按距离从小到大排序
        self.sort_index = np.argsort(self.neighbors, axis=1, kind='quicksort', order=None) 
        # 3.取前k个近邻
        self.sort_index = self.sort_index[:, 0:self.k] 
        # 4.确定前K个点所在类别的出现频率
        # 5.返回前K个点中出现频率最高的类别
        if mode == "classification":   # 分类
            return self.index2label(self.sort_index)
        if mode == "regression":       # 回归
            return self.index2value(self.sort_index)


        






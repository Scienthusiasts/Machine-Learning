import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和验证集
from sklearn.preprocessing import MinMaxScaler

import sys; sys.path.append('../')
from decisionTree import decisionTree
from myutils import readFile, createPlot

import sys
sys.setrecursionlimit(100000) #



# 读取数据集
X = np.load('npy/Cora_X.npy', allow_pickle=True)
y = np.load('npy/Cora_y.npy', allow_pickle=True)
X = np.delete(X, 444, 1) # 无用的维度，仅有一个属性
label = np.arange(X.shape[1])



# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=False)
# 划分训练集验证集
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
print(X_train.shape, X_test.shape)


DT = decisionTree(X_train, y_train, X_train, y_train, label, "gain_ratio")
# DT.initDTree()
# DT.saveTree('./digitsTree(gain_ratio).txt')
DT.loadTree('./CORATree2.txt')
# createPlot(DT.DTree)
print(DT.DTree)




acc = 0
for i in range(X_test.shape[0]):
    pred = DT.predict(DT.DTree, X_test[i,:], label) == y_test[i]
    acc += pred
print(acc/X_test.shape[0])
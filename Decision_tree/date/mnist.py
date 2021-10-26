import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和验证集
from sklearn.preprocessing import MinMaxScaler

import sys; sys.path.append('../')
from decisionTree import decisionTree
from myutils import readFile, createPlot





# 读取数据集
X = np.load('date_sets.npy')
y = np.load('date_labels.npy')
# 标准归一化
scaler = MinMaxScaler()
X = (scaler.fit_transform(X) * 4).astype(np.int)


# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 划分训练集验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)
print(X_train.shape, X_val.shape, X_test.shape)





label = np.array(['flying', 'game', 'icecream'])
DT = decisionTree(X_train, y_train, X_val, y_val, label, "gain")
DT.initDTree()
DT.saveTree('./digitsTree(gain_ratio).txt')
DT.loadTree('./digitsTree(gain_ratio).txt')
# createPlot(DT.DTree)
# print(DT.DTree)




acc = 0
for i in range(X_test.shape[0]):
    pred = DT.predict(DT.DTree, X_test[i,:], label) == y_test[i]
    acc += pred
print(acc/X_test.shape[0])


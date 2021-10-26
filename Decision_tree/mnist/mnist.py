import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.append('../')
from decisionTree import decisionTree
from myutils import readFile, createPlot





# 读取数据集
X_train = np.load('./npy/train_sets.npy').reshape(-1, 28*28)
y_train = np.load('./npy/train_labels.npy')
X_test = np.load('./npy/valid_sets.npy').reshape(-1, 28*28)
y_test = np.load('./npy/valid_labels.npy')


# X_train[X_train>0]=1
# X_test[X_test>0]=1


# X_train //= 50
# X_test //= 50


# 数据集可视化
for i in range(32):
    plt.subplot(4, 8, i+1)
    img = X_train[i,:].reshape(28, 28)
    plt.imshow(img)
    plt.title(y_train[i])
    plt.axis("off")                
    plt.subplots_adjust(hspace = 0.3)  # 微调行间距
plt.show()



label = np.array([i for i in range(28*28)])

print(X_train.shape,X_test.shape)

DT = decisionTree(X_train[:50000,:], y_train[:50000], X_train[50000:60000,:], y_train[50000:60000],label, "gain_ratio")
# DT.initDTree()
# DT.saveTree('./digitsTree(gain_ratio).txt')
# print(DT.DTree)
DT.loadTree('./digitsTree(gain_ratio).txt')
# createPlot(DT.DTree)
print(DT.DTree)

X = X_test
y = y_test

acc = 0
for i in range(10000):
    pred = DT.predict(DT.DTree, X[i,:], label) == y[i]
    acc += pred
print(acc/10000)


import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.append('../')
from decisionTree import decisionTree
from myutils import readFile, createPlot



# 读取数据集
X_train = np.load('X_train.npy').reshape(1934, 1024)
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy').reshape(946, 1024)
y_test = np.load('y_test.npy')
label = np.array([i for i in range(1024)])
print(X_train.shape,X_test.shape)



# 数据集可视化
for i in range(32):
    plt.subplot(4, 8, i+1)
    img = X_train[i*60,:].reshape(32, 32)
    plt.imshow(img)
    plt.title(y_train[i*60])
    plt.axis("off")                
    plt.subplots_adjust(hspace = 0.3)  # 微调行间距
plt.show()



DT = decisionTree(X_train, y_train, X_train, y_train, label, "gain_ratio")
DT.initDTree()
DT.saveTree('./digitsTree(gain_ratio).txt')
print(DT.DTree)
DT.loadTree('./digitsTree(gain_ratio).txt')
createPlot(DT.DTree)




acc = 0
for i in range(946):
    X = X_test[i,:]
    acc += DT.predict(DT.DTree, X, label) == y_test[i]
print(acc/946)

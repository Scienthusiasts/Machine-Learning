import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets  # 数据集模块
from sklearn.model_selection import train_test_split  # 划分训练集和验证集

from kNN import kNN
# 导入自定义评估模块:
import sys; sys.path.append('../metrics')
from metrics import metrics



# 读取数据集
X, y = datasets.load_digits(return_X_y=True)
# 划分训练集和验证集,使用sklearn中的方法
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# KNN最近邻进行分类
knn = kNN(200, X_train, y_train, X_test)
pred = knn.kNN()
# 分类准确率
accuracy = np.mean(pred == y_test)
print(pred.shape)
print('准确率:', accuracy)


label = [0,1,2,3,4,5,6,7,8,9]
# 绘制混淆矩阵
metrics.confusion_matrix_vis(y_test, pred, label)
precision, recall = metrics.precision_recall(y_test, pred, 10)
for i in range(10):
    print('类别%d: 查准率:%f, 召回率:%f' % (i, precision[i], recall[i]))
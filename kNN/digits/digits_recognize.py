import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集和验证集

# 导入自定义评估模块:
import sys; sys.path.append('../')
from kNN import kNN
sys.path.append('../../metrics')
from metrics import metrics



# 读取数据集
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')


# 数据集可视化
for i in range(32):
    plt.subplot(4, 8, i+1)
    img = X_train[i*60,:]
    plt.imshow(img)
    plt.title(y_train[i*60])
    plt.axis("off")                
    plt.subplots_adjust(hspace = 0.3)  # 微调行间距
plt.show()



# KNN最近邻进行分类
knn = kNN(3, X_train, y_train, X_test)
pred = knn.kNN()
# 分类准确率
accuracy = np.mean(pred == y_test)
print(pred.shape)
print('准确率:', accuracy)


# label = [0,1,2,3,4,5,6,7,8,9]
# # 绘制混淆矩阵
# metrics.confusion_matrix_vis(y_test, pred, label)
# metrics.precision_recall(y_test, pred, 10)

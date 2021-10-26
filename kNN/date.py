import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.preprocessing import StandardScaler # 标准归一化
from sklearn.model_selection import train_test_split  # 划分训练集和验证集

from kNN import kNN

# 读取约会数据集
def read_datasets(path):
    X, y = [], []

    file = open(path)
    for line in file.readlines():
        data = line.split('\t')
        X.append([float(data[i]) for i in range(3)])
        y.append(int(data[3]))

    return np.array(X), np.array(y)


def draw(X, y):
    # 数据集3D可视化
    fig = plt.figure()
    # 3D绘图
    ax = fig.add_subplot(111, projection='3d')
    # 按类别分类
    X_sort = [np.where(y==i+1) for i in range(3)]
    color = ["red", "green", "blue"]
    label = ["不喜欢", "一般", "极具魅力"]
    for i in range(3):
        ax.scatter(X[X_sort[i], 0], X[X_sort[i], 1], X[X_sort[i], 2], s=5, c=color[i], label=label[i]) 
        ax.legend()
    ax.set_ylim(0,80000)
    ax.set_zlim(0,80000)
    plt.show()





if __name__ == '__main__':
    path = './datingTestSet2.txt'
    # 读取数据
    X, y = read_datasets(path)

    # 划分训练集验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)





    # KNN 最近邻进行分类
    knn = kNN(5, X_train, y_train, X_test)
    pred = knn.kNN()
    # 分类准确率
    accuracy0 = np.mean(pred == y_test)
    print('准确率:', accuracy0)




    # 标准归一化
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    # 数据集3D可视化
    # draw(X_train, y_train)

    
    # KNN 最近邻进行分类
    knn = kNN(5, X_train, y_train, X_test)
    pred = knn.kNN()
    # 分类准确率
    accuracy = np.mean(pred == y_test)
    # print(pred.shape)
    print('准确率:', accuracy)
    print(accuracy0/accuracy)


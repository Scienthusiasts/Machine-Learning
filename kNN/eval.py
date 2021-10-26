import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets  # 数据集模块
from sklearn.model_selection import train_test_split  # 划分训练集和验证集

from kNN import kNN
from datasets import datasets

# kNN预测并保存结果
def save_prediction():
    # 读取数据集
    data = datasets()
    X_grid, Y_grid, Z, X, y = data.gen_data()
    # 划分训练集和验证集,使用sklearn中的方法
    # KNN最近邻进行分类
    knn = kNN(50, X, y, X)
    pred = knn.kNN(mode="regression").reshape(50, 50)
    np.save('k=1.npy', pred)

# 可视化kNN回归效果
def k_regression_visualize():
    k = ['1', '50', '300', '500', '1000', '2000', '2300', '2500']
    data = datasets()
    X_grid, Y_grid, Z, _, _ = data.gen_data()

    plt.figure(figsize=(26, 13))
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.title('k='+k[i])
        pred = np.load('k='+k[i]+'.npy')
        contour = plt.contourf(X_grid, Y_grid, pred, 100, cmap='bwr')
        plt.colorbar(contour)
    plt.subplots_adjust(left=0.02,bottom=0.05,right=0.98,top=0.95,wspace=0.07,hspace=0.1)
    plt.savefig('./kNN.png',dpi=100)
    # plt.show()

# 绘制k-acc曲线
def k_acc():
    acc = np.load('kNN_k_acc.npy')
    plt.plot(acc)
    plt.title('k-acc curve')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()





save_prediction()
# k_regression_visualize()
# k_acc()

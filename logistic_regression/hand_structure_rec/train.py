import sklearn.datasets as datasets # 数据集模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练集和验证集
import sklearn.metrics # sklearn评估模块
from sklearn.preprocessing import StandardScaler # 标准归一化
from sklearn.metrics import accuracy_score


import sys;sys.path.append('../')
from logistic import logistic


if __name__ == "__main__":
    # 设置超参数
    LR= 1e-5         # 学习率
    EPOCH = 2000   # 最大迭代次数
    BATCH_SIZE = 200  # 批大小

    # 导入数据集
    X1 = np.load("./hand_dist/cloth_dist.npy")
    y1 = np.zeros(X1.shape[0])
    X2 = np.load("./hand_dist/stone_dist.npy")
    y2 = np.ones(X1.shape[0])
    X  = np.concatenate((X1,X2), axis=0).reshape(-1,21*21)
    y  = np.concatenate((y1,y2), axis=0)

    model = logistic(X, y, mode="train")

    model.train(LR, EPOCH, BATCH_SIZE)
    model.save()

    idx = 635
    print(model.eval(X[idx,:]), y[idx])

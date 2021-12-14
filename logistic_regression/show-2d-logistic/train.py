import numpy as np
import matplotlib.pyplot as plt

import generate_classification_datasets as cf # 二维数据集
import sys;sys.path.append('../')
from logistic import logistic





if __name__ == "__main__":
    # 设置超参数
    LR= 1e-6         # 学习率
    EPOCH = 2000   # 最大迭代次数
    BATCH_SIZE = 100  # 批大小

    # 导入自己的数据集
    dataSet = cf.classification()
    X, y = dataSet.BinarySample2D(point_num=5000)
    plt.scatter(X[:,0], X[:,1],s=2,c=y)
    plt.show()

    model = logistic(X, y, mode="train")

    model.train(LR, EPOCH, BATCH_SIZE)
    # model.save()



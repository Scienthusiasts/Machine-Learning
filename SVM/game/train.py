import numpy as np    
import random

import sys;sys.path.append('../../metrics')
import sys;sys.path.append('../')
from SVM import SVM




if __name__ == '__main__':


    # # 导入数据集(手势关键点)
    cloth_X = np.load("./cloth_dist.npy")
    stone_X = np.load("./stone_dist.npy")
    scissors_X = np.load("./scissors_dist.npy")
    y = np.ones(cloth_X.shape[0])
    # one verse one 
    X0  = np.concatenate((cloth_X,stone_X), axis=0).reshape(-1,21*21)
    X1  = np.concatenate((cloth_X,scissors_X), axis=0).reshape(-1,21*21)
    X2  = np.concatenate((stone_X,scissors_X), axis=0).reshape(-1,21*21)
    y  = np.concatenate((y,-y), axis=0).reshape(-1,1)
    # three mixed together 
    X012 = np.concatenate((X0,scissors_X.reshape(-1,21*21)), axis=0).reshape(-1,21*21)
    y012 = np.concatenate((np.zeros(cloth_X.shape[0]),np.ones(cloth_X.shape[0])), axis=0)
    y012 = np.concatenate((y012,np.ones(cloth_X.shape[0])*2), axis=0)
    print(X012.shape, y012.shape)

    kernelType = 0
    Xs = {0:X0, 1:X1, 2:X2}
    for i in Xs.keys():
        svm = SVM( Xs[i], y, C=600, ε=1e-4, ker=kernelType)
        # W, b = svm.train(maxIter=1)
        # svm.save_weight_lin('./W&b_'+str(i)+'.npy')

    ''' 测试 '''
    # 读取权重
    weight0 = np.load('./W&b_0.npy')
    W0, b0 = weight0[:-1], weight0[-1]
    weight1 = np.load('./W&b_1.npy')
    W1, b1 = weight1[:-1], weight1[-1]
    weight2 = np.load('./W&b_2.npy')
    W2, b2 = weight2[:-1], weight2[-1]
    # 测试(三个模型)
    res0 = svm.linear_eval(X012, W0, b0).reshape(-1)>0
    res1 = svm.linear_eval(X012, W1, b1).reshape(-1)>0
    res2 = svm.linear_eval(X012, W2, b2).reshape(-1)>0

    # 三分类
    pred = []
    for i in range(len(res0)):
        if(res0[i] and res1[i]):pred.append(0)
        elif(not res0[i] and res2[i]):pred.append(1)
        elif(not res1[i] and not res2[i]):pred.append(2)
        else:pred.append(-1)

    acc = sum(pred==y012) / X012.shape[0]
    print("准确率: %f" % (acc))

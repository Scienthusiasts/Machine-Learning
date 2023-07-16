import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import matplotlib.pyplot as plt

root = "./mnist"
img_series = []
for file in os.listdir(root):
    img = np.array(Image.open(os.path.join(root, file)).convert('L'))
    img_series.append(img.reshape(-1))

# 注意这里每一张图像是矩阵里的一个列向量
imgs = np.r_[img_series]

X = imgs
print(X.shape)

#数据标准化，去均值
X_std = StandardScaler().fit_transform(X)
#构造协方差矩阵
Cov_mat = np.cov(X_std.T)
#使用Numpy包的linalg.eig模块求解特征值，特征向量
eig_vals,eig_vects = np.linalg.eig(Cov_mat)
eig_vects = eig_vects.real
#将特征值与特征向量组合成元组，并按照特征值进行排序
eig_pairs = [((np.abs(eig_vals[i])),eig_vects[:,i]) for i in range(len(eig_vects))]


#选取前三个特征向量构成特征矩阵
matrix_w = np.hstack((eig_pairs[0][1].reshape(-1,1),eig_pairs[1][1].reshape(-1,1)))

#将原始数据映射到新的特征矩阵空间中
Y = X.dot(matrix_w)
print(Y.shape)
plt.scatter(Y[:,0],Y[:,1], s=1)
plt.show()

import os
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# 加载数据(图像)
# 输入：文件夹根目录
# 输出：图像矩阵(行向量存储) [数据个数, 数据维度]
def loadData(root):
    dataset = []
    label = []
    print('加载数据集...')
    for classes in tqdm(os.listdir(root)):
        for imgfile in os.listdir(os.path.join(root, classes)):
            img = np.array(Image.open(os.path.join(root, classes, imgfile)).convert('L'))
            dataset.append(img.reshape(-1) / 255.)
            label.append(int(classes))
    dataset = np.array(dataset)
    return dataset, np.array(label)


# PCA降维算法
# 输入：数据矩阵(每一行为一个数据)
# 输出：降维后数据矩阵(每一行为一个数据)
def ED_PCA(X, k):
    # 注意这里每一张图像是矩阵里的一个列向量[数据维度, 数据个数]
    X = X.T
    # 计算协方差矩阵(或使用np.cov())
    # 每个维度0均值化
    X = X - np.mean(X, axis=1).reshape(-1,1)
    CovM = (X @ X.T) / X.shape[1] # X.shape[1]相当于矩阵的系数，除不除都无所谓，特征向量最后都会归一化处理
    # 对协方差矩阵进行特征分解(相似对角化)[求解的结果由于近似可能包含复数，此时只取实部]
    # ξ 中每一列是一个特征向量
    λ, ξ = np.linalg.eig(CovM)
    P = ξ[:, :k].real.T 
    # 将原始d维数据映射到k=2维
    Y = P @ X
    print(P.shape, X.shape)
    return Y.T


# SVD求解PCA
# 输入：数据矩阵(每一行为一个数据)
# 输出：降维后数据矩阵(每一行为一个数据)
def SVD_PCA(X, k):
    X = X.T
    # 每个维度0均值化
    X = X - np.mean(X, axis=1).reshape(-1,1)
    # 注意svd输入为每一列为一个数据[数据维度, 数据个数]
    U, Σ, VT = np.linalg.svd(X)
    print('U.shape, Σ.shape, VT.shape:', U.shape, Σ.shape, VT.shape)
    print(X.shape, U[:, :k].shape)
    Y = X.T @ U[:, :k]
    return Y.T


# 可视化三个维度(每一行为一个数据)
def visulize3D(Y, label):
    print(Y.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    cmap = plt.cm.get_cmap("Spectral")
    scatter = ax.scatter(Y[0,:],Y[1,:], Y[2,:], s=2, c=label, cmap=cmap)
    # 在图中显示每个类别对应的颜色(标签)
    legend = ax.legend(*scatter.legend_elements(),loc="best", title="Classes")
    ax.add_artist(legend)
    plt.show()


# 可视化两个维度(每一行为一个数据)
def visulize2D(Y, label):
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap("Spectral")
    scatter = ax.scatter(Y[0,:],Y[1,:], s=1, c=label, cmap=cmap)
    legend = ax.legend(*scatter.legend_elements(),loc="best", title="Classes")
    ax.add_artist(legend)
    plt.show()


if __name__ == "__main__":
    root = 'F:/DataSets(no used yet)/Mnist_jpg/jpg/test'
    dataset, label = loadData(root)
    # 使用特征分解(ED)求解PCA
    # Y = ED_PCA(dataset, 2)
    # visulize2D(Y.T, label)
    # 使用奇异值分解(SVD)求解PCA
    Y = SVD_PCA(dataset,3)
    visulize3D(Y, label)

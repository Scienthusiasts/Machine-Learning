from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


# 核函数
def K(xi, xj, param=0):
    # print(xi.shape, xj.shape)
    if param == 0:
        xj = xj.reshape(-1, 1)
        return np.dot(xi, xj)
    if param == 1: 
        σ = 1.3
        deltaRow = xi - xj
        ker = np.exp(np.dot(deltaRow, deltaRow.T) / (-2 * σ * σ))
        return ker

def eval(train_X, test_X, train_y, b, α, ker):
    res = []
    for j in range(test_X.shape[0]):
        res.append(b + np.sum([α[i] * train_y[i] * K(train_X[i,:], test_X[j,:], ker) for i in range(train_X.shape[0])]))
    return np.array(res)

# 可视化模型分类
def visualModel2D(data_X, data_y, b, α, ker=0):
    s = 30
    # 获取数据集坐标范围:
    x_min, x_max = min(data_X[:,0]), max(data_X[:,0])
    y_min, y_max = min(data_X[:,1]), max(data_X[:,1])
    x, y = np.linspace(x_min, x_max, s), np.linspace(y_min, y_max, s)
    # 把x,y数据生成mesh网格状的数据
    X_grid, Y_grid = np.meshgrid(x, y)
    # 生成标准格式数据集
    X = np.c_[X_grid.reshape(-1, 1), Y_grid.reshape(-1, 1)]
    
    pred = eval(data_X, X, data_y, b, α, ker)
    pred = pred.reshape(s, s)


    # plt.figure(figsize=(26, 13))
    plt.title('Non-Linear SVM')
    contour = plt.contourf(X_grid, Y_grid, pred, 50, cmap='RdBu')
    plt.colorbar(contour)
    # 训练集可视化
    plt.scatter(data_X[:, 0], data_X[:, 1], s=7, c=data_y, cmap=ListedColormap(['#FF0000','#0000FF']))
    # 支持向量可视化
    s_vec = np.where(α>0)[0]
    print("支持向量个数：", len(s_vec))
    plt.plot(data_X[s_vec, 0], data_X[s_vec, 1], marker='o', markersize=5, markeredgecolor='black', linestyle='none', markerfacecolor='none')

    plt.subplots_adjust(left=0.02,bottom=0.05,right=0.98,top=0.95,wspace=0.07,hspace=0.1)
    # plt.savefig('./kNN.png',dpi=100)
    plt.show()
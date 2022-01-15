from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # 混淆矩阵
import matplotlib.pyplot as plt
from collections import Counter
# import pandas as pd
# import seaborn as sns
import random


# 评估方法
class metrics():

    '''混淆矩阵可视化'''
    # y_hat.shape = [datasize,]
    # y.shape = [datasize,]
    # label.shape = [classes,]
    @staticmethod 
    def confusion_matrix_vis(y, y_hat, label):
        conf_mat = confusion_matrix(y, y_hat)
        # print(conf_mat)
        df_cm = pd.DataFrame(conf_mat, index = label, columns = label)
        heatmap = sns.heatmap(df_cm, annot = True, fmt = 'd', cmap = "seismic")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 50, ha = 'right')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.show()


    '''评估分类模型的查准率与召回率'''
    @staticmethod 
    def precision_recall(y, y_hat, label):
        classes = len(label)
        conf_mat = confusion_matrix(y, y_hat)
        # total_num = np.sum(conf_mat)

        TP = [conf_mat[i,i] for i in range(classes)]
        FP = [np.sum(conf_mat[:,i]) - TP[i] for i in range(classes)]
        FN = [np.sum(conf_mat[i,:]) - TP[i] for i in range(classes)]
        # TN = [total_num - FN[i] - FP[i] - TP[i] for i in range(classes)]

        precision = [TP[i] / (TP[i] + FP[i]) for i in range(classes)]
        recall = [TP[i] / (TP[i] + FN[i]) for i in range(classes)]
        
        # 绘制 precision recall 条形图:
        width = 0.35
        move = width / 2
        axis = np.arange(classes)
        plt.bar(axis - move, precision, width=width, label = "precision")
        plt.bar(axis + move, recall, width=width, label = "recall")
        for x, p,r in zip(axis, precision,recall):
            plt.text(x - move,p,"%.2f" % p,ha='center',va='bottom')
            plt.text(x + move,r,"%.2f" % r,ha='center',va='bottom')
        plt.legend()
        # 更改坐标轴：axis->label
        plt.xticks(axis, label)
        plt.show()

        return precision, recall





def visualDataSets2D(X, y):
    # # 获取标签中的类别
    # classes = dict(Counter(y))
    # # 为不同的类别添加不同的颜色
    # for i in classes:
    #     random.seed(i)
    #     classes[i] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    plt.scatter(X[:, 0], X[:, 1], s=7, c=y,  cmap=ListedColormap(['#FF0000','#0000FF']))
    plt.show()

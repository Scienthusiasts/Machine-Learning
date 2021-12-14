import numpy as np  
from naiveBayes import naiveBayes
import matplotlib.pyplot as plt
from tqdm import trange # 进度条库
from sklearn.preprocessing import StandardScaler # 标准归一化

import sys;sys.path.append('../metrics')
from metrics import metrics

datapath = 'MNIST'
X_train = np.load('../datasets/%s/train_sets.npy' % datapath).reshape(-1,784)
y_train = np.load('../datasets/%s/train_labels.npy' % datapath)
X_test = np.load('../datasets/%s/valid_sets.npy' % datapath).reshape(-1,784)
y_test = np.load('../datasets/%s/valid_labels.npy' % datapath)

# 取对数使得数据近似为高斯分布(并不是。。)
# X_train = np.log(X_train.astype(np.float) + 1)
# X_test = np.log(X_test.astype(np.float) + 1)
# 归一化
# X_train = X_train / 255
# X_test = X_test / 255
# 二值化
# X_train[X_train>0]=1
# X_test[X_test>0]=1
# 数据集可视化
for i in range(32):
    plt.subplot(4, 8, i+1)
    img = X_train[i,:].reshape(28, 28)
    plt.imshow(img)
    plt.title(y_train[i])
    plt.axis("off")         
    plt.subplots_adjust(hspace = 0.3)  # 微调行间距
plt.show()




NB = naiveBayes(X_train, y_train)
# 模型训练:
NB.calcLikelihood()
NB.saveWeight("weight_gau.txt")
# 导入"权重":
NB.loadWeight('weight_gau.txt',10)

# 计算测试集精度：
pred = []
sum = 0
for i in trange(1000):
    pred.append(NB.calcPosterior(X_test[i,:]))
    sum += pred[-1] == y_test[i]
print(sum / 1000)
np.save('pred_gau.npy',np.array(pred))


# pred = np.load('pred_gau.npy')
# sum = 0
# for i in trange(10000):
#     sum += pred[i] == y_test[i]
# print(sum / 10000)
# # 绘制混淆矩阵:
# label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# metrics.confusion_matrix_vis(y_test, pred, label)
# metrics.precision_recall(y_test, pred, label)

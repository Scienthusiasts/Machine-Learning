import sklearn.datasets as datasets # 数据集模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练集和验证集
import sklearn.metrics # sklearn评估模块
from sklearn.preprocessing import StandardScaler # 标准归一化
from sklearn.metrics import accuracy_score


# import sys
# sys.path.append('..')
# from generateDataSets import generate_classification_datasets as cf

# 导入自己的数据集
# dataSet = cf.classification()
# X, y = dataSet.BinarySample2D(point_num=5000)
# plt.scatter(X[:,0], X[:,1],s=2,c=y)
# plt.show()

# 设置超参数
LR= 1e-5         # 学习率
EPOCH = 20000   # 最大迭代次数
BATCH_SIZE = 200  # 批大小
THRESHOLD = 1e-6 # 判断收敛条件

# 导入数据集
X, y = datasets.load_breast_cancer(return_X_y=True) #乳腺癌二分类数据集



r= X.shape[0]
y = y.reshape(-1,1)
# 标准归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)


X = np.concatenate((np.ones((r, 1)), X), axis=1)
# 划分训练集和验证集,使用sklearn中的方法
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
m, n = X_train.shape[0], X_train.shape[1]
# 每个epoch包含的批数
NUM_BATCH = m//BATCH_SIZE+1
print(X_train.shape[0])
# 1. 随机初始化W参数
# 初始化参数太大会导致损失可能太大导致上溢出
W = np.random.rand(n, 1) * 0.1 # 均匀分布size = [n,1]范围为[0,1]


train_loss = []
test_loss = []
train_acc = []
test_acc = []


# 将概率转化为预测的类别
def binary(y):
    y = y > 0.5
    return y.astype(int)


def sigmoid(x):
    return 1/(1 + np.exp(-x))



# 二分类交叉熵损失
def cross_entropy(y_true, y_pred):
    # y_pred太接近1会导致后续计算np.log(1 - y_pred) = -inf
    crossEntropy = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / (y_true.shape[0])
    return crossEntropy


for i in range(EPOCH):

    # 这部分代码打乱数据集，保证每次小批量迭代更新使用的数据集都有所不同
    # 产生一个长度为m的顺序序列
    index = np.arange(m)
    # shuffle方法对这个序列进行随机打乱
    np.random.shuffle(index)
    # 打乱
    X_train = X_train[index]
    y_train =y_train[index]
    # 在验证集上评估:
    pred_y_test = sigmoid(np.dot(X_test, W))
    test_loss.append(cross_entropy(y_true=y_test, y_pred=pred_y_test))
    test_acc.append(accuracy_score(y_true=y_test, y_pred=binary(pred_y_test)))
    # 在训练集上评估:
    pred_y_train = sigmoid(np.dot(X_train, W))
    train_loss.append(cross_entropy(y_true=y_train, y_pred=pred_y_train))
    train_acc.append(accuracy_score(y_true=y_train, y_pred=binary(pred_y_train)))

    if i % 1000 == 0:
        print("eopch: %d | train loss: %.6f | test loss: %.6f | train acc.:%.4f | test acc.:%.4f" % (i, train_loss[i], test_loss[i], train_acc[i], test_acc[i]))

    for batch in range(NUM_BATCH-1):
        # 切片操作获取对应批次训练数据(允许切片超过列表范围)
        X_batch = X_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        y_batch = y_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

        # 2. 求梯度,需要用上多元线性回归对应的损失函数对W求导的导函数
        previous_y = sigmoid(np.dot(X_batch, W))
        grad = np.dot(X_batch.T, previous_y - y_batch)

        # 加入正则项
        # grad = grad + np.sign(W) # L1正则
        grad = grad + W            # L2正则

        # 3. 更新参数,利用梯度下降法的公式
        W = W - LR * grad


# 打印最终结果
for loop in range(32):
    print('===', end='') 
print("\ntotal iteration is : {}".format(i))
y_hat_train = sigmoid(np.dot(X_train, W))
loss_train = cross_entropy(y_true=y_train, y_pred=y_hat_train)
print("train loss:{}".format(loss_train))
y_hat_test = sigmoid(np.dot(X_test, W))
loss_test = cross_entropy(y_true=y_test, y_pred=y_hat_test)
print("test loss:{}".format(loss_test))
print("train acc.:{}".format(train_acc[-1]))
print("test acc.:{}".format(test_acc[-1]))

# # 保存权重
# np.save("Weight.npy",W)
# np.save("train_loss.npy",train_loss)
# np.save("test_loss.npy",test_loss)
# np.save("train_acc.npy",train_acc)
# np.save("test_acc.npy",test_acc)

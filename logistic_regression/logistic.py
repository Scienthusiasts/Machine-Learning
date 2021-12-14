import sklearn.datasets as datasets # 数据集模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练集和验证集
import sklearn.metrics # sklearn评估模块
from sklearn.preprocessing import StandardScaler # 标准归一化
from sklearn.metrics import accuracy_score



# 设置超参数
LR= 1e-5         # 学习率
EPOCH = 20000   # 最大迭代次数
BATCH_SIZE = 200  # 批大小


class logistic:
    def __init__(self, X=None, y=None, mode="pretrain"):
        self.X = X
        self.y = y
        # 记录训练损失，准确率
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
        # 标准归一化
        self.scaler = StandardScaler()
        # 1. 初始化W参数
        if mode == "pretrain":
            self.W = np.load("../eval_param/Weight.npy")
        else:
            # 随机初始化W参数 初始化参数太大会导致损失可能太大,导致上溢出
            self.W = np.random.rand(X.shape[1]+1, 1) * 0.1 # 均匀分布size = [n,1]范围为[0,1]

    # 将概率转化为预测的类别
    def binary(self, y):
        y = y > 0.5
        return y.astype(int)


    # 二分类sigmoid函数
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))



    # 二分类交叉熵损失
    def cross_entropy(self, y_true, y_pred):
        # y_pred太接近1会导致后续计算np.log(1 - y_pred) = -inf
        crossEntropy = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / (y_true.shape[0])
        return crossEntropy

    # 数据预处理(训练)
    def train_data_process(self):
        self.y = self.y.reshape(-1,1)
        # 划分训练集和验证集,使用sklearn中的方法
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        # 标准归一化
        self.std =  np.std(X_train, axis=0) + 1e-8
        self.mean = np.mean(X_train,axis=0)
        X_train = (X_train - self.mean) / self.std
        X_test = (X_test - self.mean) / self.std
        # 加入偏置
        X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
        X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
        # 每个epoch包含的批数
        NUM_BATCH = X_train.shape[0]//BATCH_SIZE+1
        print('训练集大小:', X_train.shape[0], '验证集大小:', X_test.shape[0])
        return X_train, X_test, y_train, y_test, NUM_BATCH



    def test_data_process(self, X):
        X = X.reshape(1,-1)
        # 标准归一化
        self.std =  np.load('../eval_param/std.npy')
        self.mean = np.load('../eval_param/mean.npy')
        X = (X - self.mean) / self.std
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X




    def train(self, LR, EPOCH, BATCH_SIZE):
        X_train, X_test, y_train, y_test, NUM_BATCH = self.train_data_process()
        for i in range(EPOCH):

            # 这部分代码打乱数据集，保证每次小批量迭代更新使用的数据集都有所不同
            # 产生一个长度为m的顺序序列
            index = np.arange(X_train.shape[0])
            # shuffle方法对这个序列进行随机打乱
            np.random.shuffle(index)
            # 打乱
            X_train = X_train[index]
            y_train = y_train[index]
            # 在验证集上评估:
            pred_y_test = self.sigmoid(np.dot(X_test, self.W))
            self.test_loss.append(self.cross_entropy(y_true=y_test, y_pred=pred_y_test))
            self.test_acc.append(accuracy_score(y_true=y_test, y_pred=self.binary(pred_y_test)))
            # 在训练集上评估:
            pred_y_train = self.sigmoid(np.dot(X_train, self.W))
            self.train_loss.append(self.cross_entropy(y_true=y_train, y_pred=pred_y_train))
            self.train_acc.append(accuracy_score(y_true=y_train, y_pred=self.binary(pred_y_train)))

            # 可视化决策边界(针对二维数据集)
            # if (i+1) % 30 == 0:
            #     self.plot_2D_line(X_train, y_train)

            if i == 0 or (i+1) % BATCH_SIZE == 0:
                print("eopch: %d | train loss: %.6f | test loss: %.6f | train acc.:%.4f | test acc.:%.4f" % 
                (i+1, self.train_loss[i], self.test_loss[i], self.train_acc[i], self.test_acc[i]))

            for batch in range(NUM_BATCH-1):
                # 切片操作获取对应批次训练数据(允许切片超过列表范围)
                X_batch = X_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
                y_batch = y_train[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

                # 2. 求梯度,需要用上多元线性回归对应的损失函数对W求导的导函数
                previous_y = self.sigmoid(np.dot(X_batch, self.W))
                grad = np.dot(X_batch.T, previous_y - y_batch)

                # 加入正则项
                # grad = grad + np.sign(W) # L1正则
                grad = grad + self.W            # L2正则

                # 3. 更新参数,利用梯度下降法的公式
                self.W = self.W - LR * grad


        # 打印最终结果
        for loop in range(32):
            print('===', end='') 
        print("\ntotal iteration is : {}".format(i+1))
        y_hat_train = self.sigmoid(np.dot(X_train, self.W))
        loss_train = self.cross_entropy(y_true=y_train, y_pred=y_hat_train)
        print("train loss:{}".format(loss_train))
        y_hat_test = self.sigmoid(np.dot(X_test, self.W))
        loss_test = self.cross_entropy(y_true=y_test, y_pred=y_hat_test)
        print("test loss:{}".format(loss_test))
        print("train acc.:{}".format(self.train_acc[-1]))
        print("test acc.:{}".format(self.test_acc[-1]))



    def eval(self, X):
        X = self.test_data_process(X)
        y_hat = self.sigmoid(np.dot(X, self.W))
        return y_hat
        



    # 保存权重
    def save(self):
        np.save("../eval_param/Weight.npy", self.W)
        np.save("../eval_param/std.npy", self.std)
        np.save("../eval_param/mean.npy", self.mean)

        np.save("../eval_param/train_loss.npy", self.train_loss)
        np.save("../eval_param/test_loss.npy", self.test_loss)
        np.save("../eval_param/train_acc.npy", self.train_acc)
        np.save("../eval_param/test_acc.npy", self.test_acc)

    # 可视化决策边界(针对二维数据集)
    def plot_2D_line(self, X_train, y_train):
        min = [np.min(X_train[:,1]), np.min(X_train[:,1])]
        max = [np.max(X_train[:,2]), np.max(X_train[:,2])]
        w = self.W.reshape(-1)
        x = np.array([min[0],max[0]])
        y = (-w[0]-w[1]*x) / w[2]
        plt.scatter(X_train[:,1], X_train[:,2],s=2,c=y_train.reshape(-1))
        plt.xlim(min[0],max[0])
        plt.ylim(min[1],max[1])
        plt.plot(x,y)
        plt.pause(0.01)
        plt.cla()

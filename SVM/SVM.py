import numpy as np      
import sys;sys.path.append('../metrics')
# from metrics import visualDataSets2D
import generate_classification_datasets as cf # 二维数据集
import random

from visual import visualModel2D

# 读取文本数据
def loadDataSet(filename):
    file = open(filename, 'r')
    X = []
    y = []
    for sample in file.readlines():
        data = sample.split('\t')
        X.append([float(data[0]), float(data[1])])
        y.append(float(data[2]))

    return np.array(X), np.array(y)


# 数据集用到的参数统一定义一个数据结构进行存储:
class SVM:
    def __init__(self, X, y, C, ε, ker):
        self.X = X                # 数据集
        self.y = y                # 标签
        self.C = C                # 松弛变量在损失中的权重
        self.ε = ε                # 容忍误差
        self.m = X.shape[0]       # X总数
        self.α = np.zeros((self.m, 1)) # 待求解参数α
        self.b = 0  # 待求解参数b
        self.ker = ker # 核函数参数
        # eCache存储每个参数的误差E，第一列为是否有效的标志位(是否计算过而不是初始的0)
        self.eCache = np.zeros((self.m, 2)) 




    # 根据约束条件约束求得的α(矩形+线性区域)
    def clipAlpha(self, αj, H, L):
        if αj > H:
            αj = H
        if αj < L:
            αj = L
        return αj


    # 核函数
    def K(self, xi, xj, param=0):
        if param == 0:
            xj = xj.reshape(-1, 1)
            return np.dot(xi, xj)
        if param == 1: 
            σ = 1.3
            deltaRow = xi - xj
            ker = np.exp(-np.dot(deltaRow, deltaRow.T) / (2 * σ * σ))
            return ker




    # 计算误差Ek
    def calcEk(self, k):
        # 计算SVM模型的预测结果：
        fk = np.sum([self.α[i] * self.y[i] * self.K(self.X[i,:], self.X[k,:], self.ker) for i in range(self.m)]) + self.b
        # 计算误差:
        Ek = fk.reshape(1) - self.y[k]
        return Ek


    # 更新第k个参数的误差
    def updateEk(self, k):
        # 计算误差
        Ek = self.calcEk(k)
        # 更新计算后的误差
        self.eCache[k,0] = 1
        self.eCache[k,1] = Ek

    # 随机选取第二个参数, 不同于i即可
    def selectJRand(self, i, m):
        j = i
        while(j==i):
            j = int(random.uniform(0, m))
        return j
    
    # 启发式选取第j个参数, 更高效
    def selectJ(self, i, Ei):
        bestJ = -1     # 最佳误差对应第几个α 
        maxDeltaE = 0  # 记录最大误差|Ei - Ej|
        Ej = 0         # 记录最佳参数αj对应的误差Ej
        # 获取那些已经更新过的Ek(非0)
        validE = np.nonzero(self.eCache[0,:])[0]
        if len(validE) > 1:
            # 遍历那些已经计算过的Ei，选择其中最大的对于的αj作为第二个参数
            for k in validE:
                # j和i不能是同一个
                if k == i: continue
                # 计算当前alpha的E
                Ek = self.calcEk(k)
                # 两个差值要最大
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    bestJ = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return bestJ, Ej
        else:   # 对于第一次更新的情况，Ek还没更新过
            j = self.selectJRand(i, self.m)
            Ej = self.calcEk(j)
        return j, Ej

    # SMO优化算法流程:
    def innerL(self, i):
        # 计算误差
        Ei = self.calcEk(i)
        # K.K.T.条件  (ε是一个很小的数，容忍一些误差？)
        # (1) αi = 0     =>  yi(wTxi + b) - 1 > 0 + ε 【非支持向量】
        # (2) 0 < αi < C =>  yi(wTxi + b) - 1 = 0     【支持向量】
        # (3) αi = C     =>  yi(wTxi + b) - 1 ≤ 0 - ε 【内嵌向量】

        # 启发式寻找参数i(如果满足K.K.T.条件直接跳过)：
        # if语句用来判断是否违法了K.K.T.条件(3)(1):
        if ((self.y[i]*Ei < -self.ε) and (self.α[i] < self.C)) or ((self.y[i]*Ei > self.ε) and (self.α[i] > 0)):
            # 启发式寻找参数j：
            j, Ej = self.selectJ(i, Ei)
            old_αi, old_αj = self.α[i].copy(), self.α[j].copy()

            # 先求好这三个参数，之后会用到
            K11 = self.K(self.X[i,:], self.X[i,:], self.ker)
            K12 = self.K(self.X[i,:], self.X[j,:], self.ker)
            K22 = self.K(self.X[j,:], self.X[j,:], self.ker)

            # αi 和 αj 的解带有约束:
            # (1) αiyi + αjyj = ζ
            # (2) 0 < αi,j < C
            # 分情况求解待约束的解：
            if self.y[i] != self.y[j]:
                L = max(0, self.α[j] - self.α[i])
                H = min(self.C, self.C + self.α[j] - self.α[i])
            else:
                L = max(0, self.α[j] + self.α[i] - self.C)
                H = min(self.C, self.α[j] + self.α[i])
            
            # 打印一些优化时的参数信息
            # L == H的情况是αj正好在矩形约束对角线的点上
            # 若 L = H，αj = L = H
            if(L == H): print("出现 L = H = %f, i = %d, j = %d, αj=%f" %(L, i, j, self.α[j])); return 0
            η = -2.0 * K12 + K11 + K22
            # 若 η <= 0，就无需优化了？？？
            if η <= 0: print("η <= 0"); return 0
            # 若 η >= 0,有更新公式：
            self.α[j] += self.y[j] * (Ei - Ej) / η
            # 约束α的范围
            self.α[j] = self.clipAlpha(self.α[j], H, L)
            # 更新αj的误差
            self.updateEk(j)
            if(np.abs(self.α[j] - old_αj) < 1e-5):
                print("j 更新的太少")
                return 0
            # 更新αi
            self.α[i] += self.y[i] * self.y[j] * (old_αj - self.α[j])
            # 接下来求解参数b:
            D_αi, D_αj = (self.α[i] - old_αi), (self.α[j] - old_αj)
            b1 = - Ei - self.y[i] * K11 * D_αi - self.y[j] * K12 * D_αj + self.b
            b2 = - Ej - self.y[i] * K12 * D_αi - self.y[j] * K22 * D_αj + self.b
            # 分情况更新参数b：
            if 0 < self.α[i] < self.C: self.b = b1
            elif 0 < self.α[j] < self.C: self.b = b2
            else: self.b = (b1 + b2) / 2.0
            return 1
        else: return 0

    # 训练流程，基于SMO算法
    def train(self, maxIter):
        iter = 0 # 记录迭代次数
        entireSet = True 
        αPairsChanged = 0
        # 当迭代次数超过最大迭代次数或任何α都无需优化时(收敛), 退出
        while(iter < maxIter) and ((αPairsChanged > 0) or (entireSet)):
            αPairsChanged = 0
            if entireSet:
                print("===========================全数据集遍历===========================")
                # 遍历针对所有向量
                for i in range(self.m):
                    # 选取αij并更新
                    αPairsChanged += self.innerL(i)
                    print("Iteration:%d | choosed i=%d | 已更新的参数对数:%d" % (iter, i, αPairsChanged))
                iter += 1
                # print("已优化所有参数，算法提前结束！")
                # break
            else: 
                print("===========================非边界值遍历===========================")
                # 仅遍历非边界向量
                nonBoundIs = np.nonzero((self.α>0) * (self.α<self.C))[0]
                # 遍历非边界值
                for i in nonBoundIs:
                    αPairsChanged += self.innerL(i)
                    print("Iteration:%d | choosed i=%d | 已更新的参数对数:%d" % (iter, i, αPairsChanged))
                iter += 1
            # 遍历一次后改为非边界遍历
            if entireSet: 
                entireSet = False
            # 如果alpha没有更新，计算全样本遍历
            elif (αPairsChanged == 0):
                entireSet = True

        if(iter >= maxIter):print("超过最大迭代次数，算法结束")
        if not((αPairsChanged < 0) or (entireSet)):print("算法收敛")

        # 对于线性核:
        if(self.ker == 0):
            self.W = np.zeros((self.X.shape[1], 1))
            for i in range(self.m):
                self.W += self.α[i] * self.y[i] * self.X[i,:].reshape(-1,1)
            return self.W, self.b, self.α
        # 对于非线性核:
        return self.b, self.α

    # 保存权重(线性核)
    def save_weight_lin(self, path):
        weight = np.concatenate((self.W, self.b.reshape(-1,1)), axis=0)
        np.save(path, weight)


    # 测试
    # 核心公式就是 y = wTx + b
    # 在超平面上方时为y>0,为类别1
    # 在超平面下方时为y<0,为类别-1
    def eval(self, test_X):
        res = []
        sv_index = np.nonzero(self.α)[0]
        for j in range(test_X.shape[0]):
            res.append(self.b + np.sum([self.α[i] * self.y[i] * self.K(self.X[i,:], test_X[j,:], self.ker) for i in sv_index]))
        return np.array(res)

    # 对于线性核，直接采用y = wTx + b判别方法，避免计算需要使用X(映射到高维空间的核数据不行)
    @staticmethod
    def linear_eval(test_X, W, b):
        res = b + np.dot(test_X, W)
        return res










if __name__ == '__main__':
    kernelType = 1

    # 读取数据(txt)
    X, y = loadDataSet('./testSet.txt')
    y = y.reshape(-1,1)
    # visualDataSets2D(X, y)

    svm = SVM( X, y, C=600, ε=1e-4, ker=kernelType)
    b, α = svm.train(maxIter=1)
    # print(α)
    # 测试
    res = svm.eval(X)
    acc = sum(res*y>0) / X.shape[0]
    print("准确率: %f" % (acc))
    visualModel2D(X, y, b, α, kernelType)


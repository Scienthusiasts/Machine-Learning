import numpy as np  
import pickle 
from collections import Counter




class naiveBayes:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.cls_num = []

    '''读取权重'''
    def loadWeight(self, path, cls_num):
        file = open(path, 'rb')
        weight = pickle.load(file)

        if len(weight) == cls_num:
            self.likelihood = weight
        else:
            self.likelihood = weight[:-1]
            self.cls_num = weight[-1]

    '''保存权重'''
    def saveWeight(self, path):
        file = open(path, 'wb')
        if len(self.cls_num) == 0:
            pickle.dump(self.likelihood, file)
        else:
            pickle.dump(self.likelihood + [self.cls_num], file)

        
    # '''伯努利朴素贝叶斯(0,1);多元伯努利朴素贝叶斯(0,1,...,n)'''
    # '''计算似然概率'''
    # # 适用于二值化图像(离散值处理)
    # def calcLikelihood(self, N):
    #     cls_num = []
    #     total_num = self.X.shape[0]
    #     # 统计数据集中的所有类别
    #     classes = set(self.y)
    #     # 将同类的数据划分到一起
    #     division = [self.X[np.where(self.y==i)] for i in classes]
    #     # 计算先验概率 = 该类别数 / 数据集总数
    #     self.prior = [type.shape[0] / total_num for type in division]
    #     # 计算似然概率：
    #     likelihood = np.zeros((len(classes), self.X.shape[1]))
    #     # 转化为列表方便添加不同类型的元素(字典)
    #     likelihood = likelihood.tolist()
    #     # 计算似然概率
    #     for i, type in enumerate(division):
    #         for j in range(self.X.shape[1]):
    #             # 统计属性出现次数
    #             tmp_dict = dict(Counter(type[:,j]))
    #             # 似然概率 = 属性出现次数 / 数据集这个类别的总数
    #             for key in tmp_dict.keys():
    #                 tmp_dict[key] = (tmp_dict[key] + 1) / (type.shape[0] + N)
    #             likelihood[i][j] = tmp_dict
    #         cls_num.append(type.shape[0])
    #     self.likelihood = likelihood
    #     self.cls_num = cls_num



    # '''计算后验概率'''
    # # 适用于二值化图像(离散值处理)
    # def calcPosterior(self, X_test, N):
    #     cls_num = len(self.likelihood)
    #     sum = np.ones(cls_num)
    #     # 逐维度计算似然概率，条件概率=似然概率的乘积(每个维度独立的假设下)
    #     for i in range(cls_num):
    #         for j in range(X_test.shape[0]):
    #             if X_test[j] in self.likelihood[i][j].keys():
    #                 # 取对数运算防止下溢出
    #                 sum[i] += np.log(self.likelihood[i][j][X_test[j]]) 
    #                 # sum[i] *= self.likelihood[i][j][X_test[j]]
    #             else:
    #                 # laplacian修正，防止概率为0
    #                 sum[i] += np.log(1 / (self.cls_num[i] + N)) 
    #                 # sum[i] *= 1e-3
    #     return np.argmax(sum)


    '''多项式朴素贝叶斯'''
    '''计算似然概率(适用于文本分类)'''
    # 统计某类别下的词频
    def calcWordFreq(self, bow, texts):
        hist = np.zeros(len(bow))
        for i, text in enumerate(texts):
            for word in text:
                if word in bow:
                    hist[bow.index(word)] += 1
            print(i)
        return hist


    '''计算后验概率(适用于文本分类 no unique/ unique)'''
    def calcPosterior(self, X, freq_vec, bow):
        sum = np.zeros(freq_vec.shape[0])
        # 统计不同类别下的后验概率：
        for i in range(freq_vec.shape[0]):
            total_num = np.sum(freq_vec[i,:])
            X_cnt = dict(Counter(X))
            for word in X_cnt.keys():
                # 如果词语在词袋中:
                if word in bow: 
                    # 统计词频
                    frec = freq_vec[i, bow.index(word)] / total_num
                    if frec != 0:
                        sum[i] += X_cnt[word] * np.log(frec)
                        # print(X_cnt[word])
                    # 如果词频向量中该词语=0
                    else:
                        # 平滑一个小概率，概率不能为 0
                        sum[i] += np.log(1 / total_num)
                # 如果词语不在词袋中:
                else:
                    # 平滑一个小概率，概率不能为 0
                    sum[i] += np.log(1 / total_num)
        return np.argmax(sum)


    

    # '''高斯朴素贝叶斯'''
    # # 适用于一般图像(连续值处理, 使用正态分布)
    # def calcLikelihood(self):
    #     total_num = self.X.shape[0]
    #     # 统计数据集中的所有类别
    #     classes = set(self.y)
    #     # 将同类的数据划分到一起
    #     division = [self.X[np.where(self.y==i)] for i in classes]
    #     # 计算先验概率 = 该类别数 / 数据集总数
    #     self.prior = [type.shape[0] / total_num for type in division]
    #     # 计算似然概率：
    #     likelihood = np.zeros((len(classes), self.X.shape[1]))
    #     # 转化为列表方便添加不同类型的元素(字典)
    #     likelihood = likelihood.tolist()
    #     # 计算似然概率
    #     for i, type in enumerate(division):
    #         for j in range(self.X.shape[1]):
    #             μ = np.mean(type[:,j])
    #             σ = np.var(type[:,j])
    #             likelihood[i][j] = [μ, σ]
    #     self.likelihood = likelihood




    # # 假设连续值服从正态分布
    # def gaussian_distribution(self, x, μ, σ):
    #     σ += 1e-1   # 平滑
    #     return np.exp(-(x - μ)*(x - μ) / (2 * σ)) / (np.sqrt(2 * np.pi * σ) )

    # def calcPosterior(self, X_test):
    #     cls_num = len(self.likelihood)
    #     sum = np.zeros(cls_num)
    #     for i in range(cls_num):
    #         for j in range(X_test.shape[0]):
    #             μ = self.likelihood[i][j][0]
    #             σ = self.likelihood[i][j][1]
    #             if X_test[j] == 0 and σ == 0:
    #                 sum[i] -= 5 # 防止加上一个过大的值
    #             else:
    #                 sum[i] += np.log(self.gaussian_distribution(X_test[j], μ, σ))
    #     return np.argmax(sum)

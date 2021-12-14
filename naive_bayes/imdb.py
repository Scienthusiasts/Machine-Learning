import numpy as np  
import pickle
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler # 标准归一化
from naiveBayes import naiveBayes
from tqdm import trange # 进度条库


file = open('imdb_cut/train_neg_cut.txt', 'rb');X_train_neg = pickle.load(file)
file = open('imdb_cut/train_pos_cut.txt', 'rb');X_train_pos = pickle.load(file)
file = open('imdb_cut/test_neg_cut.txt', 'rb');X_test_neg = pickle.load(file)
file = open('imdb_cut/test_pos_cut.txt', 'rb');X_test_pos = pickle.load(file)
file = open('imdb_cut/bag.txt', 'rb');bow = pickle.load(file)

# 模型训练:
NB = naiveBayes(X_train_neg, X_train_neg)
# 统计正负样本的词频：
neg_hist = NB.calcWordFreq(bow, X_train_neg)
pos_hist = NB.calcWordFreq(bow, X_train_pos)
np.save('neg_hist.npy', neg_hist)
np.save('pos_hist.npy', pos_hist)
# 导入训练好的词频:
# neg_hist = np.load('imdb_cut/neg_hist.npy').reshape(1,-1)
# pos_hist = np.load('imdb_cut/pos_hist.npy').reshape(1,-1)
# freq_vec = np.concatenate((neg_hist, pos_hist), axis=0)
# # 词频可视化
# plt.subplot(311)
# plt.plot(neg_hist[0])
# plt.ylim(0,6000)
# plt.xlabel('neg Words index')
# plt.ylabel('frequency')
# plt.subplot(312)
# plt.plot(pos_hist[0])
# plt.ylim(0,6000)
# plt.xlabel('pos Words index')
# plt.ylabel('frequency')
# plt.subplot(313)
# plt.plot((pos_hist[0]+1) / (neg_hist[0]+1),marker='o')
# plt.plot(np.ones(len(bow)),color='r')
# plt.xlabel('Words index')
# plt.ylabel('frequency ratio')
# plt.legend()
# plt.show()


# # 评估测试集:
# sum_pos = []
# sum_neg = []

# for i in trange(12500):
#     pred = NB.calcPosterior(X_test_pos[i], freq_vec, bow) == 1
#     # print(i, pred)
#     sum_pos.append(pred)
# for i in trange(12500):
#     pred = NB.calcPosterior(X_test_neg[i], freq_vec, bow) == 0
#     # print(i, pred)
#     sum_neg.append(pred)
    
# # pred = np.load('naive_bayes/imdb_cut')
# print(np.sum(sum_pos) / 12500)
# print(np.sum(sum_neg) / 12500)
# # 保存测试结果:
# np.save('pred_pos.npy', np.array(sum_pos))
# np.save('pred_neg.npy', np.array(sum_neg))

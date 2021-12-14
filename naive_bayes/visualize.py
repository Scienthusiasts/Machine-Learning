import numpy as np  
import matplotlib.pyplot as plt 
import pickle 
from collections import Counter


'''基于高斯朴素贝叶斯的似然概率最大化'''
def show_ber(likelihood):
    for step, image in enumerate(likelihood):
        img = np.zeros(len(image))
        for i in range(len(image)):
            # 遍历所有属性
            for attr in image[i].keys():
                # 所有属性以概率值加权和作为灰度值:
                img[i] += attr * image[i][attr]
        plt.subplot(2,5,step+1)
        plt.imshow(img.reshape(28, 28))
        plt.axis("off")              
        plt.title(step) 
    plt.show()



'''基于高斯朴素贝叶斯的似然概率最大化'''
def show_gau(likelihood):
    for step, image in enumerate(likelihood):
        img = np.zeros(len(image))
        for i in range(len(image)):
            img[i] = image[i][0]
        plt.subplot(2,5,step+1)
        plt.imshow(img.reshape(28, 28))
        plt.axis("off")              
        plt.title(step)  
        plt.subplots_adjust(hspace = 0.3)  # 微调行间距
    plt.show()

def trans(X):
    return (X + 128) % 256

def show_distribution():
    X_train = np.load('../datasets/MNIST/train_sets.npy')
    X_train = trans(X_train).reshape(-1)
    np.save('trans.npy', X_train)
    # X_train = np.load('count_trans.npy')
    count = np.zeros(256)
    for i in X_train:
        count[i] +=1
    axis = np.arange(256)
    plt.bar(axis, count, width=1)
    plt.xlabel("grey scale")
    plt.ylabel("frequency")
    plt.show()


if __name__ == '__main__':

    # 可视化贝叶斯似然概率模型
    # file = open('weight_bin.txt', 'rb')
    # likelihood = pickle.load(file)[:-1]
    # show_ber(likelihood)

    show_distribution()


import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# import cupy as np
from tqdm import tqdm, trange
plt.rcParams[ 'font.sans-serif' ]=['simHei'] # 用来正常显示中文标签

# 单张图像压缩(复原到图像的秩为k)
def rank(k, U, Σ, VT):
    Σ = np.diag(Σ[:k])
    return U[:, :k] @ Σ @ VT[:k, :]

# 可视化单张复原后图像
# k值表示要复原到图像的秩为k
def show_single_picture(k, U, Σ, VT, mode=0, img=None):
    if mode == 0:
        plt.figure(figsize=(12,12))
        for i in range(1, k*k+1):
            img = rank(i, U, Σ, VT)

            plt.subplot(k, k, i)
            plt.imshow(img,cmap='gray')
            plt.axis("off")
            plt.subplots_adjust(hspace=0.1,wspace=0.1, left=0.01, right=0.99, bottom=0.01, top=0.99)
        plt.show()
    # 只要秩=mode的结果
    else:
        img = rank(mode, U, Σ, VT)
        return img



# 可视化一张图像的SVD矩阵
def show_single_img_svd(img, U, Σ, VT):
    plt.subplot(141)
    plt.imshow(img,cmap='gray')
    plt.title('raw image')
    plt.subplot(142)
    plt.imshow(U,cmap='gray')
    plt.title('U')
    plt.subplot(143)
    plt.imshow(np.diag(Σ),cmap='gray')
    plt.title('Σ')
    plt.subplot(144)
    plt.imshow(VT,cmap='gray')
    plt.title('V.T')
    plt.show()



# 可视化右奇异向量(可以理解为所有图像的成分)(每一张图像是矩阵里的一个行向量)
def show_Principal(k):
    _, _, VT = loadSVD()
    plt.figure(figsize=(12,12))
    for i in range(1, k*k+1):

        plt.subplot(k, k, i)
        plt.imshow(VT[i].reshape(64,64),cmap='gray')
        plt.axis("off")
        plt.subplots_adjust(hspace=0.1,wspace=0.1, left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.savefig('./result/右奇异向量(图像成分)',dpi=150)
    plt.show()


# 可视化图像成分加权合成一张图像
def show_single_principal(No, k, img):
    U, Σ, VT = loadSVD()
    plt.imshow(img[No,:].reshape(64,64),cmap='gray')

    plt.figure(figsize=(12,12))
    weight = (U[No,:Σ.shape[0]] * Σ).reshape(1,-1)
    r = 1
    for i in trange(1, k*k+1):
        plt.subplot(k, k, i)
        img = weight[:, :r] @ VT[:r, :]
        r += 3
        plt.imshow(img.reshape(64,64),cmap='gray')
        plt.axis("off")
        plt.subplots_adjust(hspace=0.1,wspace=0.1, left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.savefig('./result/图像成分加权合成单张图像(%d)'%(No),dpi=150)
    plt.show()




# 加载单张图像
def load_image(root, mode='gray'):
    if mode=='gray':
        img = Image.open(root).convert('L')
    else:
        img = Image.open(root)
    return np.array(img) 


# 对一张图像做SVD分解(提取的是一张图像的特征)
def singleImgSVD(img):
    # 奇异值分解(img = UΣVᵀ)
    U, Σ, VT = np.linalg.svd(img)
    print('U, Σ, VT size:', U.shape, Σ.shape, VT.shape)
    return U, Σ, VT



# 加载图像数据集
def load_datasets(root, num):
    img_series = []
    print('读取图像数据集...')
    for file in tqdm(os.listdir(root)[:num]):
        img = np.array(Image.open(os.path.join(root, file)).convert('L'))
        img_series.append(img.reshape(-1))
    img_series = np.array(img_series)
    # 注意这里每一张图像是矩阵里的一个行向量
    print('数据集大小:[数据个数, 数据维度]', img_series.shape)
    return img_series

def loadSVD():
    U = np.load('SVD_weight/U.npy')
    Σ = np.load('SVD_weight/Σ.npy')
    VT = np.load('SVD_weight/VT.npy')
    return U, Σ, VT




# 对若干图像做SVD分解(提取的是这类图像的普遍特征)
# 输入：根目录地址; 输出：图像矩阵的 U, Σ, VT
def multiImgsSVD(imgs, save=False):
    # 奇异值分解(img = UΣVᵀ)
    U, Σ, VT = np.linalg.svd(imgs)
    print('U, Σ, VT size:', U.shape, Σ.shape, VT.shape)
    np.save('SVD_weight/U.npy',U)
    np.save('SVD_weight/Σ.npy',Σ)
    np.save('SVD_weight/VT.npy',VT)
    return U, Σ, VT


if __name__ == '__main__':
    root = "./resized/"
    imgs = load_datasets(root, 10000)
    U, Σ, VT = multiImgsSVD(imgs, save=True)
    show_Principal(10)
    

# if __name__ == '__main__':
#     # root = "F:/DataSets(no used yet)/Mnist_jpg/jpg/test/5/356.jpg"
#     root = "./resized/0.jpg"
#     img = load_image(root)
#     U, Σ, VT = singleImgSVD(img)
#     show_single_picture(8, U, Σ, VT)
#     show_single_img_svd(img, U, Σ, VT)


# if __name__ == '__main__':
#     root = './4.png'
#     r = 200
#     img = load_image(root, mode="RGB")
#     imgr, imgg, imgb = img[:,:,0],img[:,:,1],img[:,:,2]
#     U_r, Σ_r, VT_r = singleImgSVD(imgr)
#     U_g, Σ_g, VT_g = singleImgSVD(imgg)
#     U_b, Σ_b, VT_b = singleImgSVD(imgb)
#     limgr = show_single_picture(6, U_r, Σ_r, VT_r, mode=r, img=imgr)
#     limgg = show_single_picture(6, U_g, Σ_g, VT_g, mode=r, img=imgg)
#     limgb = show_single_picture(6, U_b, Σ_b, VT_b, mode=r, img=imgb)
#     image = np.stack((limgr, limgg, limgb), axis=2)/255

#     plt.subplot(121)
#     plt.imshow(img)  
#     plt.title('raw image,size = %d x %d'%(img.shape[0], img.shape[1]))  
#     plt.subplot(122)
#     plt.title('low rank image(rank=%d)\n图像压缩率=%f'%(r, (img.shape[0]+img.shape[1]+1)*r/(img.shape[0]*img.shape[1])))  
#     plt.imshow(image)
#     plt.subplots_adjust(hspace=0.1,wspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.95)
#     plt.show()


import matplotlib.pyplot as plt    
import cv2
import numpy as np  

# 绘制随时间变化下的手势数据集
def draw(X, ax, step):
    lines = []
    lines.append(X[[0,1,2,3,4]])
    lines.append(X[[0,5,6,7,8]])
    lines.append(X[[0,17,18,19,20]])
    lines.append(X[[5,9,13,17]])
    lines.append(X[[9,10,11,12]])
    lines.append(X[[13,14,15,16]])
    for line in lines:
        ax.plot(line[:,0], line[:,1], line[:,2], linewidth=2)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(-0.5,0.5)
    # plt.savefig("./visualize_hand/%d.png" % step) 
    plt.pause(0.01)
    plt.cla()

# 三维手势数据集可视化
def show_frames(hand_frame):
    fig = plt.figure()
    # 3D绘图
    ax = fig.add_subplot(111, projection='3d')
    for step, hand in enumerate(hand_frame):
        draw(hand, ax, step)

# 计算手势关键点一点和所有点之间的距离
def calc_dist(p1, p2):
    x = (p1[0]-p2[:,0]) * (p1[0]-p2[:,0])
    y = (p1[1]-p2[:,1]) * (p1[1]-p2[:,1])
    z = (p1[2]-p2[:,2]) * (p1[2]-p2[:,2])
    dist = x + y + z
    return dist

def distance(X):
    dataSets = []
    for step, hand in enumerate(X):
        dist_matrix = []
        for i in range(21):
            dist_matrix.append(calc_dist(hand[i], hand))
        # plt.imshow(dist_matrix)
        # plt.savefig("./visualize_hand/%d.png" % step) 
        # plt.pause(0.01)
        # plt.cla()
        dataSets.append(dist_matrix)
    return np.array(dataSets)






if __name__ == "__main__":
    hand_frames = np.load('./hand_frame/hand_frame_cloth.npy')
    # show_frames(hand_frames)


    dataSets = distance(hand_frames)
    # np.save('./hand_dist/one_dist.npy',dataSets)
    # print(dataSets.shape)


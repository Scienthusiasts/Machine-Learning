import numpy as np   
import os 
from PIL import Image

def encode_img(path):
    img_sets = []
    for i in range(10000):
        image = str(i) + '.png'
        img = np.array(Image.open(path + image))
        img_sets.append(img)

    img_sets = np.array(img_sets)
    np.save('valid_sets.npy' ,img_sets)


def encode_label(path):
    file = open(path)
    line = file.readline().split(',')
    line = [int(i) for i in line]
    line = np.array(line)
    print(line.shape)
    np.save('valid_labels.npy' ,line)



imgpath = 'valid/'
labelpath = 'valid_label.txt'
encode_img(imgpath)
# encode_label(labelpath)
import os
import numpy as np
import matplotlib.pyplot as plt



def txt2img(path):
    X, y = [], []
    for files in os.listdir(path):
        file = open(path + files)
        data = []
        for line in file.readlines():
            row = []
            for pix in line[:-1]:
                row.append(int(pix))
            data.append(np.array(row))
        X.append(np.array(data))
        y.append(int(files.split('_')[0]))

    return np.array(X), np.array(y)




path = './testDigits/'
X, y = txt2img(path)
np.save('X_test.npy', X)
np.save('y_test.npy', y)
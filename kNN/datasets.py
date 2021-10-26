import numpy as np
import matplotlib.pyplot as plt

class datasets():

    def F(self, x, y):
        return np.sin(np.sqrt(x**2 + y**2)) + np.cos(x)

    def gen_data(self):
        # 生成x,y的数据
        n = 50
        x, y = np.linspace(-30, 30, n), np.linspace(-30, 30, n)
        # 把x,y数据生成mesh网格状的数据
        X, Y = np.meshgrid(x, y)
        Z = self.F(X, Y)

        x = X.reshape(-1, 1) 
        y = Y.reshape(-1, 1)
        z = Z.reshape(-1)
        data = np.c_[x, y]
        
        return X, Y, Z, data, z




if __name__ == '__main__':
    # 填充等高线
    data = datasets()
    X, Y, Z, data, _ = data.gen_data()
    print(data)
    contour = plt.contourf(X, Y, Z, 100)
    plt.colorbar(contour)
    plt.show()

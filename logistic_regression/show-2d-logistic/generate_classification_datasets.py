import numpy as np  


# 分类数据集
class classification():


    def randSample(self, mu=0, scale=1, num=100):
        return np.random.normal(loc=mu, scale=scale, size=(num,1))
        
    # 生成2分类数据集
    def BinarySample2D(self, point_num=500):
        # 加入噪声
        point_num //=2
        x0_x = self.randSample(mu=-3, scale=1.2, num=point_num)
        x0_y = self.randSample(mu=-1, scale=0.8, num=point_num)
        x0 = np.concatenate((x0_x,x0_y),1)
        y0 = np.zeros(point_num)

        x1_x = self.randSample(mu=0, scale=1.3, num=point_num)
        x1_y = self.randSample(mu=0, scale=0.7, num=point_num)
        x1 = np.concatenate((x1_x,x1_y),1)
        y1 = np.ones(point_num)

        X = np.concatenate((x0,x1),0)
        y = np.concatenate((y0,y1),0)
        
        return X, y


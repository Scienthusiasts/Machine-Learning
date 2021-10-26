import numpy as np       
import pandas as pd



def load_Cora():
    dataSet=pd.read_csv('cora.content',sep = '\t',header=None)
    feature = dataSet.iloc[:,1:]
    X = np.array(feature)[:, :-1]
    y = np.array(feature)[:, -1]
    return X, y



classes = {'Case_Based':0, 'Genetic_Algorithms':1, 'Neural_Networks':2,'Probabilistic_Methods':3, 
          'Reinforcement_Learning':4, 'Rule_Learning':5, 'Theory':6}

X, y = load_Cora()
y = [classes[i] for i in y]
np.save('Cora_X.npy',X)
np.save('Cora_y.npy',y)

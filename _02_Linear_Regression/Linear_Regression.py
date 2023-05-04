# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x, y = read_data()
    x = np.insert(x, 0, 1, axis=1)
    lam = 0.01
    w = np.linalg.inv(x.T.dot(x) + lam * np.identity(x.shape[1])).dot(x.T).dot(y)
    return np.dot(data, w[1:]) + w[0]
    
def lasso(data):
    x, y = read_data()
    x = np.insert(x, 0, 1, axis=1)
    lam = np.exp(-12)
    w = np.zeros(x.shape[1])
    alpha = 0.01
    for i in range(15000):
        w[1:] -= alpha * (x[:, 1:].T.dot(x[:, 1:].dot(w[1:]) - y) + lam * np.sign(w[1:]))
        w[0] -= alpha * (x[:, 0].T.dot(x[:, 0].dot(w[0]) - y))
    return np.dot(data, w[1:]) + w[0]

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

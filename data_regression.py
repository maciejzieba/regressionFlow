import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

kk = 3


def w(x):
    return np.log(1-x/kk)/np.log(2/kk)


def n_w(x):
    return np.floor(w(x))


def r(x):
    return w(x) - n_w(x)


def WW(x):
  n = n_w(x)
  # print(n)
  if n==0:
    return np.array([r(x)])
  else:
    return np.hstack([np.ones( int(n) ), np.array([r(x)])])


def ZZ(x):
  n = n_w(x)
  sign = np.random.choice([-1,1], int(n) +1, replace=True, p=[0.5, 0.5])
  lists = np.array([(1/2**i) for i in range( int(n) +1 )])
  return sign * lists


def M(x):
  w = WW(x)
  z = ZZ(x)
  # print(w)
  # print(z)
  return w@z.T


def get_data3(n_size=10000):
    M_v = np.vectorize(M)
    s = np.random.uniform(0, kk, n_size)
    m = M_v(s)
    y = np.vstack((s, m)).T
    return y


def get_data(n_size=10000, mu=0.0, var=0.01):
    x = np.random.randn(n_size, 2)
    x_norm_1 = np.expand_dims(np.linalg.norm(x, 1, axis=1), 1).repeat(2, axis=1)
    x_norm_2 = np.expand_dims(np.linalg.norm(x, 2, axis=1), 1).repeat(2, axis=1)
    r = np.expand_dims(np.random.randn(n_size), 1).repeat(2, axis=1)
    x1 = x / x_norm_1
    y1 = np.exp(mu + np.sqrt(var) * r) * x1
    x2 = x / x_norm_2
    y2 = np.exp(mu + np.sqrt(var) * r) * x2
    y2 = y2 - np.asarray([0, 2])
    y = np.concatenate([y1, y2])
    return y


def get_data2(n_size=10000, mu=0.0, var=0.01):
    x = np.random.randn(n_size, 2)
    x_norm = np.expand_dims(np.linalg.norm(x, 2, axis=1), 1).repeat(2, axis=1)
    r = np.expand_dims(np.random.randn(n_size), 1).repeat(2, axis=1)
    x = x / x_norm
    y = np.exp(mu + np.sqrt(var) * r) * x
    return y


class ExampleData(Dataset):
    def __init__(self, n_size=10000, mu=0.0, var=0.01):
        self.train_points = get_data3(n_size)

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        return self.train_points[idx, 0], self.train_points[idx, 1]

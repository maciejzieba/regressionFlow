import numpy as np
from torch.utils.data import Dataset


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
    # plt.scatter(y[:, 0], y[:, 1])
    # plt.show()
    return y


def get_data2(n_size=10000, mu=0.0, var=0.01):
    x = np.random.randn(n_size, 2)
    x_norm = np.expand_dims(np.linalg.norm(x, 2, axis=1), 1).repeat(2, axis=1)
    r = np.expand_dims(np.random.randn(n_size), 1).repeat(2, axis=1)
    x = x / x_norm
    y = np.exp(mu + np.sqrt(var) * r) * x
    return y


class ExampleData(Dataset):
    def __init__(self, n_size=5000, mu=0.0, var=0.01):
        self.train_points = get_data(n_size, mu, var)

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        return self.train_points[idx, 0], self.train_points[idx, 1]

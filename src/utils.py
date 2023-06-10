import numpy as np

def mean_squared_error(y, t):
    N = y.size()
    return np.sum((y - t)**2) / (2 * N)


def cross_entropy_error(y, t, eps=1e-7):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size
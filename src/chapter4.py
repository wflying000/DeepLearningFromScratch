import os
import sys
import numpy as np
from tqdm import tqdm

from utils import softmax, sigmoid, cross_entropy_error
from mnist import load_mnist

def numerical_diff(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient_1d(f, x, h=1e-4):
    grad = np.zeros_like(x)

    for idx in tqdm(range(x.size), total=x.size, leave=False):
        tmp = x[idx]
        
        # f(..., x + h, ...)
        x[idx] = tmp + h
        y1 = f(x)

        # f(..., x - h, ...)
        x[idx] = tmp - h
        y2 = f(x)

        grad[idx] = (y1 - y2) / (2 * h)

        x[idx] = tmp # 还原当前变量值
    
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in tqdm(enumerate(X), total=len(X), leave=False):
            grad[idx] = numerical_gradient_1d(f, x)
        
        return grad

# def numerical_gradient(f, x):
#     h = 1e-4 # 0.0001
#     grad = np.zeros_like(x)
    
#     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
#     while not it.finished:
#         idx = it.multi_index
#         tmp_val = x[idx]
#         x[idx] = float(tmp_val) + h
#         fxh1 = f(x) # f(x+h)
        
#         x[idx] = tmp_val - h 
#         fxh2 = f(x) # f(x-h)
#         grad[idx] = (fxh1 - fxh2) / (2*h)
        
#         x[idx] = tmp_val # 还原值
#         it.iternext()   
        
#     return grad

def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x) # 计算f在x处的梯度
        x -= lr * grad # 更新x
    
    return x

def function2(x):
    return x[0] ** 2 + x[1] ** 2

def test_gradient_descent_function2():
    init_x = np.array([-3.0, 4.0])
    lr = 0.1
    step_num = 100
    x = gradient_descent(function2, init_x, lr, step_num)
    print(x)


class SimpleNet():
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

def test_gradient_descent_SimpleNet():
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])

    def f(W):
        return net.loss(x, t)
    
    dW = numerical_gradient(f, net.W)

    print(dW)

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
    
def generate_batch(X, t, batch_size):
    total = len(X)
    indexes = list(range(total))
    np.random.shuffle(indexes)
    for i in range(0, total, batch_size):
        index = indexes[i : i + batch_size]
        x_batch = X[index]
        t_batch = t[index]
        yield x_batch, t_batch
        


def train():
    os.chdir(sys.path[0])

    (x_train, t_train), (x_test, t_test) = load_mnist(
        filepath="../data/mnist.pkl",
        normalize=True, 
        one_hot_label=True
    )

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    network = TwoLayerNet(
        input_size=784, 
        hidden_size=50, 
        output_size=10
    )

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask] # 每次随机去batch_mask指定的索引的数据全部跑完可能还存在未取到的数据
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)


        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch {i}, train_acc: {train_acc}, test_acc: {test_acc}")

if __name__ == "__main__":
    # test_gradient_descent_function2()
    # test_gradient_descent_SimpleNet()
    train()
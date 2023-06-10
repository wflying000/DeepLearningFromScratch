import numpy as np
import matplotlib.pyplot as plt

from mnist import load_mnist

# 激活函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

def plot_step_function():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def relu(x):
    return np.maximum(0, x)

def plot_relu():
    x = np.arange(-5.0, 5.0, 0.1) 
    y = relu(x)
    plt.plot(x, y)
    plt.show()

def softmax(x):
    m = np.max(x)
    exp_x = np.exp(x - m)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def test_softmax():
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)


def identity_function(x):
    return x

# 三层神经网络构造
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def test_forward():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

# mnist图像测试
def check_mnist():
    import os, sys
    os.chdir(sys.path[0])
    filepath = "../data/mnist.pkl"
    (x_train, y_train), (x_test, y_test) = load_mnist(filepath, flatten=True, normalize=False)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)



if __name__ == "__main__":
    # plot_step_function()
    # plot_sigmoid()
    # plot_relu()
    # test_forward()
    check_mnist()
# 《深度学习入门 —— 基于Python的理论与实现》chapter4 神经网络的学习

本章主要内容
1. 两个常用损失函数：均方误差(Mean Squared Error, MSE)与交叉熵损失(CrossEntroyLoss)的实现
2. 数值微分的实现，通过数值微分计算网络参数的导数
3. 构造一个手写数字识别网络，通过数值微分实现网络的学习

神经网络的学习主要是通过训练数据与监督标签计算损失函数，然后计算损失函数相对于网络参数的偏导数，然后更新网络参数，实现网络的学习，如下所示
$$
\theta = \theta - \alpha\frac{\partial L}{\partial \theta}
$$

## 1. 损失函数
### 1.1 均方误差
对于回归任务常用均方误差MSE作为损失函数
$$
MSE = \frac{1}{2N} \sum_{i=1}^N{(y_i - t_i)^2}\tag{1}
$$
其中$y_i$表示网络输出, $t_i$表示监督标签。numpy实现如下
```Python
def mean_squared_error(y, t):
    N = y.size()
    return np.sum((y - t)**2) / (2 * N)
```

### 1.2 交叉熵损失
分类任务通常使用交叉熵损失函数
$$
loss = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^K{t_{ij}\log{y_{ij}}}\tag{2}
$$
其中$N$表示样本数量，$K$表示类别数量, $t$为真实标签的one-hot表示。$y$为网络输出。

假设有一个文本情感分类的任务，情感分为积极、消极、中立（$K = 3$），标签分别为0, 1, 2。一批训练数据有$N=4$个样本，真实标签分别为$[2, 0, 1, 0]$, 真实标签对应的one-hot表示分别为
$$
t_1: \begin{bmatrix}0 & 0 & 1\end{bmatrix} \\
t_2: \begin{bmatrix}1 & 0 & 0\end{bmatrix} \\
t_3: \begin{bmatrix}0 & 1 & 0\end{bmatrix} \\
t_4: \begin{bmatrix}1 & 0 & 0\end{bmatrix} \\
$$

模型输出结果为
$$
y_1: \begin{bmatrix}0.1 & 0.2 & 0.7\end{bmatrix} \\
y_2: \begin{bmatrix}0.8 & 0.1 & 0.1\end{bmatrix} \\
y_3: \begin{bmatrix}0.3 & 0.6 & 0.1\end{bmatrix} \\
y_4: \begin{bmatrix}0.1 & 0.1 & 0.8\end{bmatrix} \\
$$

按照以上公式计算损失
$$
loss = -\frac{1}{4}(t_{11}\log{y_{11}} + t_{12}\log{y_{12}} + t_{13}\log{y_{13}} + \\ t_{21}\log{y_{21}} + t_{22}\log{y_{22}} + t_{23}\log{y_{23}} + \\ t_{31}\log{y_{31}} + t_{32}\log{y_{32}} + t_{33}\log{y_{33}} + \\ t_{41}\log{y_{41}} + t_{42}\log{y_{42}} + t_{43}\log{y_{43}}
) \\ = -\frac{1}{4}(0*\log{0.1} + 0*\log{0.2} + 1*\log{0.7} + \\ 1*\log{0.8} + 0*\log{0.1} + 0*\log{0.1} + \\ 0*\log{0.3} + 1*\log{0.6} + 0*\log{0.1} + \\ 1*\log{0.1} + 0*\log{0.1} + 0*\log{0.8}) \\ = -\frac{1}{4}(1*log{0.7} + 1*log{0.8} + 1*\log{0.6} + 1*\log{0.1})
$$

通过以上计算可以，每个样本的网络输出值中除了真实标签对应的位置的数之外，其他位置的数其实对损失没有贡献(因为除了真实标签之外的位置在one-hot编码中为0)。因此可以直接通过标签值取网络输出的对应位置的值，例如对于样本1，它的标签值为2，可以通过这个标签值从$y_1$中直接取出对应的值$y_1[2] = 0.7$计算。对于样本2，它的标签值为0，可以直接从$y_2$取出$y_2[0] = 0.8$计算。

通过以上分析, 交叉熵损失函数可以修改为如下形式
$$
loss = -\frac{1}{N}\sum_{i=1}^N{\log{y_i[t_i]}}\tag{3}
$$
其中$y_i$表示第$i$个样本的输出，$t_i$表示第$i$个样本的真实标签。$y_i[t_i]$表示取$y_{i}$索引为$t_i$的位置的值。

交叉熵损失的numpy实现如下
```Python
def cross_entropy_error(y, t, eps=1e-7):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size
```

## 2. 数值微分
在损失函数求完后，如何计算损失函数相对于网络参数的梯度呢，本章中通过数值微分实现。
### 2.1 导数
对于单变量函数可以通过以下方法计算导数
$$
\frac{df(x)}{dx} = \lim_{h\rightarrow 0}\frac{f(x + h) - f(x - h)}{2h}
$$
当$h$取值足够小时，可以计算出函数$f(x)$在点$x$处的导数。Python实现如下
```Python
def numerical(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)
```

### 2.2 梯度
对于多变量函数，需要计算梯度。计算时保持其他变量不变，计算函数对当前变量的偏导数，所有变量的偏导数组合即为梯度。例如对于函数$f(x_1, x_2, ...)$, 计算函数在点$(x_a, x_b, ...)$处的梯度，需要分别计算函数$f(·)$在点$(x_a, x_b)$处关于$x_1, x_2, ...$偏导数$\frac{\partial f}{\partial x_1}|_{x_1=x_a}, \frac{\partial f}{\partial x_2}|_{x_2=x_b}...$。函数$f(·)$在$(x_a, x_b)$处的梯度为$(\frac{\partial f}{\partial x_1}|_{x_1=x_a}, \frac{\partial f}{\partial x_2}|_{x_2=x_b}, ... )$

偏导数的计算如下：
$$
\frac{\partial f}{\partial x_1} = \lim_{h\rightarrow0}\frac{f(x_a + h, x_b, ...) - f(x_a - h, x_b, ...)}{2h}
$$

$$
\frac{\partial f}{\partial x_2} = \lim_{h\rightarrow0}\frac{f(x_a, x_b + h, ...) - f(x_a, x_b - h, ...)}{2h}
$$

Python实现如下：
```Python
def numerical_gradient(f, x, h=1e-4):
    grad = np.zeros_like(x)

    for idx in range(x.size):
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
```

### 2.3 梯度下降
神经网络优化需要找到损失函数的最小值。函数在某一点的梯度是函数在该点上升最快的方向，沿着梯度的反方向即为函数在该点下降最快的方向。求得函数在某点的梯度后就可以沿着该梯度更新函数值，然后再求函数在更新后的梯度，如此迭代进行，即是梯度下降。python实现如下：
```Python
def gradient_descent(f, init_x, lr, step_num):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x) # 计算f在x处的梯度
        x -= lr * grad # 更新x
    
    return x
```

### 2.4 神经网络的梯度
神经网络中模型的参数通常为矩阵$\boldsymbol{W_{m\times n}}$, 损失函数$L$关于$\boldsymbol{W_{m\times n}}$的也为矩阵并且形状也为$m \times n$,即
$$
\frac{\partial L}{\partial W} = \begin{bmatrix}
    \frac{\partial L}{\partial W_{11}} & \frac{\partial L}{\partial W_{12}} & ··· & \frac{\partial L}{\partial W_{1n}} \\
    ·\\
    ·\\
    ·\\
    \frac{\partial L}{\partial W_{m1}} & \frac{\partial L}{\partial W_{m2}} & ··· & \frac{\partial L}{\partial W_{mn}}

\end{bmatrix}
$$

由于神经网络的参数通常为2维, 计算梯度的实现修改为
```Python
def numerical_gradient_1d(f, x, h=1e-4):
    grad = np.zeros_like(x)

    for idx in range(x.size):
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
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)
        
        return grad
```

## 3. 基于数值微分的神经网络实现

本节实现一个2层神经网络，并通过数值微分更新参数，实现手写数字识别

### 3.1 神经网络实现
2层神经网络Python实现如下

```Python
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
```

### 3.2 生成mini-batch数据

Python实现如下
```Python
def generate_batch(X, t, batch_size):
    total = len(X)
    indexes = list(range(total))
    np.random.shuffle(indexes)
    for i in range(0, total, batch_size):
        index = indexes[i : i + batch_size]
        x_batch = X[index]
        t_batch = t[index]
        yield x_batch, t_batch
```
书中本章没有在一个epoch中完整遍历整个训练集，每次训练都随机取batch_size个样本，共迭代10000次。数值微分计算特别慢，在我的电脑上测试batch_size=100时，跑一个batch需要2分多钟，因此跑10000次消耗太大。完整代码在[DeepLearningFromScratch
](https://github.com/wflying000/DeepLearningFromScratch)
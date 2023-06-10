# 《深度学习入门 —— 基于Python的理论与实现》chapter4 神经网络的学习

本章主要内容
1. 两个常用损失函数：均方误差(Mean Squared Error, MSE)与交叉熵损失(CrossEntroyLoss)的实现
2. 数值微分的实现，通过数值微分计算网络参数的导数
3. 构造一个手写数字识别网络，通过数值微分实现网络的学习

神经网络的学习主要是通过训练数据与监督标签计算损失函数，然后计算损失函数相对于网络参数的偏导数，然后更新网络参数，实现网络的学习，如下所示
$$
\theta = \theta - \alpha\frac{\partial L}{\partial \theta}
$$

## 1.损失函数
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

## 2.数值微分
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
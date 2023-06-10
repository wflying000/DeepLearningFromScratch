import os
import sys
import numpy as np

def numerical_diff(f, x, h=1e-4):
    return (f(x + h) - f(x - h)) / (2 * h)


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


from typing import Tuple
import numpy as np

def lagrange(f:Tuple,x:np.ndarray)->np.ndarray:
    xx,yy = f
    x = np.array(x)
    lx = len(x)
    y = np.zeros_like(x)
    for i in range(lx):
        if lx == 1:
            xi = x
        else:
            xi = x[i]
        w = np.ones_like(xx).astype(float)
        for j, xj in enumerate(xx):
            idx1 = np.where(xx-xj != 0.0)[0].astype(int)
            den = np.prod(xj-xx[idx1])
            if den:
                w[j] = 1.0/den
        if xi in xx:
            y[i] = yy[xx==xi]
        else:
            idx2 = np.where(xx-xi != 0.0)[0].astype(int)
            y_num = np.sum(w[idx2]*yy[idx2]/(xi-xx[idx2]))
            y_den = np.sum(w[idx2]/(xi-xx[idx2]))
            y[i] = y_num/y_den
    return y

def poly(f:Tuple, x:np.ndarray, d:int=2)->np.ndarray:
    xx, yy = f
    assert f[1][0] <= x[0] and f[1][-1] >= x[-1]
    assert d <= len(xx)
    y=np.zeros_like(x)
    for i,xi in enumerate(x):
        indices = np.argsort(np.abs(xi-xx))[:d]
        xl = xx[indices]
        yl = yy[indices]
        fl = (xl,yl)
        y[i] = lagrange(fl,[xi])
    return y







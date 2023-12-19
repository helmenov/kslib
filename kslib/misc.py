#%%
import numpy as np
from typing import Tuple
from scipy import fftpack as scifft

# それぞれのスペクトルの横軸をlog2する関数
def mylogx(xy:Tuple):
    xy = np.array(xy)
    lx = np.log2(xy[0].clip(1e-7))
    return lx,xy[1]

def z(x):
    m = np.mean(x)
    s = np.std(x,ddof=1)
    xx = (x-m)/s
    return xx

def myxcor(x:Tuple, y:Tuple, ti=1e-3, standardize=True, normalize=True):
    """cross correlation R_{x,y}(tau), in case
        timings(indices) for x is  differ from those for y.

    Args:
        x (Tuple): t[i], x(t[i])
        y (Tuple): t[i], y(t[i])
        ti (_type_, optional): tick interval for t. Defaults to 1e-3.
        standardize (bool, optional): return standardize. Defaults to True.

    Returns:
        correlation: tau, cor in cor(tau) := sum_i x(i-tau)*y(i)
    """
    t_begin = np.amin(np.r_[x[0],y[0]])
    lx = np.floor((np.amax(x[0]) -t_begin)/ ti + 1).astype(int)
    ly = np.floor((np.amax(y[0]) -t_begin)/ ti + 1).astype(int)
    N = lx if lx > ly else ly
    tl = lx + ly - 1
    tau = (np.arange(tl)-ly)*ti

    if normalize == True:
        zx = z(x[1])
        zy = z(y[1])
    else:
        zx = x[1]
        zy = y[1]

    gx = np.zeros(lx)
    gx[((x[0]-t_begin) / ti).astype(int)] = zx
    gx = np.r_[np.zeros(ly-1), gx]

    zy = z(y[1])
    gy = np.zeros(ly)
    gy[((y[0]-t_begin) / ti).astype(int)] = zy
    gy = np.r_[gy, np.zeros(lx-1)]

    Gx = scifft.fft(gx)
    Gy = scifft.fft(gy)
    Gxy = Gx * np.conj(Gy)
    cor = np.real(scifft.ifft(Gxy)) #/(N*ti)
    if standardize:
        tx, cx = myxcor(x,x,standardize=False)
        ty, cy = myxcor(y,y,standardize=False)
        cor /= np.sqrt(np.abs(cx[tx==0])) * np.sqrt(np.abs(cy[ty==0]))

    return tuple([tau, cor])

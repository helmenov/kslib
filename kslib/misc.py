import numpy as np
from typing import Tuple

# それぞれのスペクトルの横軸をlog2する関数
def mylogx(xy:Tuple):
    xy = np.array(xy)
    lx = np.log2(xy[0].clip(1e-7))
    return lx,xy[1]

def myxcor_xneq(x:Tuple, y:Tuple, ti=1e-3, standardize=True):
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
    tau = (np.arange(tl)-ly+1)*ti

    gx = np.zeros(lx)
    gx[((x[0]-t_begin) / ti).astype(int)] = x[1]
    gx = np.r_[np.zeros(ly-1), gx]

    gy = np.zeros(ly)
    gy[((y[0]-t_begin) / ti).astype(int)] = y[1]
    gy = np.r_[gy, np.zeros(lx-1)]

    Gx = scifft.fft(gx)
    Gy = scifft.fft(gy)
    Gxy = Gx * np.conj(Gy)
    cor = np.real(scifft.ifft(Gxy)) #/(N*ti)
    if standardize:
        tx, cx = myxcor_xneq(x,x,standardize=False)
        ty, cy = myxcor_xneq(y,y,standardize=False)
        cor /= np.sqrt(cx[tx==0]) * np.sqrt(cy[ty==0])

    return tuple([tau, cor])

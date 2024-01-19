#%%
from typing import Tuple

import numpy as np
from scipy import fftpack as scifft


def Otsu(x):
    def VarWithin(x,t):
        y = np.full_like(x,False)
        y[x>=t] = True
        N = len(x)
        n1 = np.count_nonzero(y)
        w1 = n1/N
        w0 = 1-w1

        if w0 == 0 or w1 == 0:
            return np.nan

        y1 = x[y==True]
        y0 = x[y==False]

        v0 = np.var(y0) if len(y0)>0 else 0
        v1 = np.var(y1) if len(y1)>0 else 0

        return w0 * v0 + w1 * v1

    RANGE = 44100
    t_range = range(RANGE)*np.max(x)/RANGE
    criteria_ = [VarWithin(x,t) for t in t_range]
    t = t_range[np.argmin(criteria_)] # best_threshold

    y = [True if xi > t else False for xi in x]

    return np.array(y)
# それぞれのスペクトルの横軸をlog2する関数
def mylogx(xy:Tuple):
    """(x,y) -> (l, y), l=log2(x)

    Args:
        xy (Tuple): (x,y)-data, x in N

    Returns:
        ly (Tuple): (l,y)-data, l in R
    """
    xy = np.array(xy)
    lx = np.log2(xy[0].clip(1e-7))
    return lx,xy[1]

def z(x):
    """Z-standardize

    Args:
        x : array data

    Returns:
        z : standardize array data
    """
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
    tau = (np.arange(tl)-ly+1)*ti

    # 正規化
    if normalize == True:
        zx = z(x[1])
        zy = z(y[1])
    else:
        zx = x[1]
        zy = y[1]

    # [i: 1,3,6]
    # [x: 2,10,37]
    # => [gx: 0,2,0,10,0,0,37] i.e. gx[t] == x[t] if t in i else x.mean
    gx = np.full(lx,zx.mean)  # 値のないところはmeanに
    gx[((x[0]-t_begin) / ti).astype(int)] = zx
    gy = np.full(ly,zy.mean)
    gy[((y[0]-t_begin) / ti).astype(int)] = zy

    # 巡回畳み込みFFTによる相互相関
    gx = np.r_[np.zeros(ly-1,dtype='complex'), gx]
    gy = np.r_[gy, np.zeros(lx-1,dtype='complex')]
    Gx = scifft.fft(gx)
    Gy = scifft.fft(gy)
    Gxy = Gx * np.conj(Gy)
    cor = np.real(scifft.ifft(Gxy)) #/(N*ti)

    # 標準化
    if standardize:
        tx, cx = myxcor(x,x,standardize=False)
        ty, cy = myxcor(y,y,standardize=False)
        cor /= np.sqrt(np.abs(cx[tx==0])) * np.sqrt(np.abs(cy[ty==0]))

    return tuple([tau, cor])

def logical_xcor(x:Tuple, y:Tuple, ti=1e-3, standardize=True):
    """cross correlation R_{x,y}(tau), in case
        timings(indices) for x is  differ from those for y.

    Args:
        x (Tuple): t[i], x(t[i])
        y (Tuple): t[i], y(t[i])
        ti (_type_, optional): tick interval for t. Defaults to 1e-3.
        standardize (bool, optional): return standardize. Defaults to True.

    Returns:
        correlation: tau, cor in cor(tau) := sum_i x(i-tau)*y(i)

    >>> x = np.random.binomial(1,0.1,1000)
    >>> i = 2*np.array(range(len(x)))
    >>> y = x
    >>> x = (i,x)
    >>> y = (i+77,y)
    >>> r = logical_xcor(x,y)
    >>> print(r[0][np.argmax(r[1])])
    -77.0
    """
    t_begin = np.amin(np.r_[x[0],y[0]])
    lx = np.floor((np.amax(x[0]) -t_begin)/ ti + 1).astype(int)
    ly = np.floor((np.amax(y[0]) -t_begin)/ ti + 1).astype(int)
    N = lx if lx > ly else ly
    tl = lx + ly - 1
    tau = (np.arange(tl)-ly+1)*ti

    def booleanize(a):
        # Convert Real Number Datum to Boolean {0:False,1:True} Datum
        amax = np.max(a)
        t = amax/2
        if a.dtype != 'bool':
            b = np.array([True if ai > t else False for ai in a])
        else:
            b = np.copy(a)
        return b

    def polar(x):
        # Convert q-ary datum to q-polar complex value datum. x must not be q-level.
        q = len(set(x))
        y = np.exp(1j * 2 * np.pi / q * x)
        return y

    zx = np.zeros(lx)  # 値のないところは 0
    zx[((x[0]-t_begin) / ti).astype(int)] = x[1]
    zy = np.zeros(ly)
    zy[((y[0]-t_begin) / ti).astype(int)] = y[1]

    zx = booleanize(zx)
    #zx = Otsu(zx)
    gx = polar(zx)

    zy = booleanize(zy)
    #zy = Otsu(zy)
    gy = polar(zy)

    # 巡回畳み込みFFTによる相互相関
    gx = np.r_[np.zeros(ly-1,dtype='complex'), gx]
    gy = np.r_[gy, np.zeros(lx-1,dtype='complex')]
    Gx = scifft.fft(gx)
    Gy = scifft.fft(gy)
    Gxy = Gx * np.conj(Gy)
    cor = np.real(scifft.ifft(Gxy)) / tl

    # 標準化
    if standardize:
        tx, cx = logical_xcor(x,x,standardize=False)
        ty, cy = logical_xcor(y,y,standardize=False)
        cor /= np.sqrt(np.abs(cx[tx==0])) * np.sqrt(np.abs(cy[ty==0]))

    return tuple([tau, cor])

if __name__ == '__main__':
    x = np.random.binomial(1,0.1,100)
    i = 2*np.array(range(len(x)))
    y = x
    x = (i,x)
    y = (i+77,y)
    r = logical_xcor(x,y)
    imax = np.argmax(r[1])
    print(r[0][imax], r[1][imax])


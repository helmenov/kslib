#%%
from typing import Tuple

import numpy as np
from scipy import fftpack as scifft
from kslib import binalize

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

def correlate_t(x:Tuple, y:Tuple, normalize = True):
    tx, vx = x
    ty, vy = y
    tau = list({Tx-Ty for Tx in tx for Ty in ty}) # setにした時点で重複消失
    tau = np.array(sorted(tau))
    K = len(tau)
    cor = np.zeros(K,dtype='complex')
    for k, tau_k in enumerate(tau):
        for i, Tx  in enumerate(tx):
            for j, Ty in enumerate(ty):
                if Ty == Tx - tau_k:
                    cor[k] += vx[i] * np.conj(vy[j])
    if normalize == True:
        cor_x, tau_x = correlate_t(x,x, normalize=False)
        cor_y, tau_y = correlate_t(y,y, normalize=False)
        ix = np.where(tau_x==0)[0]
        iy = np.where(tau_y==0)[0]
        cor /= np.sqrt(cor_x[ix]*np.conj(cor_y[iy]))

    return cor, tau

def convolute_t(x,Tuple, y:Tuple):
    tx, vx = x
    ty, vy = y
    tz = list({Tx+Ty for Tx in tx for Ty in ty}) # setにした時点で重複消失
    tz = np.array(sorted(tz))
    L = len(tz)
    conv = np.zeros(L,dtype='complex')
    for k, tz_k in enumerate(tz):
        for i, Tx  in enumerate(tx):
            for j, Ty in enumerate(ty):
                if Ty == tz_k - Tx:
                    conv[k] += vx[i] * np.conj(vy[j])
    return conv, tz


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

    #zx = booleanize(zx)
    zx = binalize.Otsu(zx)
    gx = polar(zx)

    #zy = booleanize(zy)
    zy = binalize.Otsu(zy)
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

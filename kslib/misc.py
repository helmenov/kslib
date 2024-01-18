#%%
import numpy as np
from typing import Tuple
from scipy import fftpack as scifft

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
    # => [gx: 0,2,0,10,0,0,37] i.e. gx[t] == x[t] if t in i else 0
    gx = np.zeros(lx, dtype='complex')
    gx[((x[0]-t_begin) / ti).astype(int)] = zx
    gy = np.zeros(ly, dtype='complex')
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
    def booleanize(a):
        if a.dtype != 'bool':
            b = np.array([True if ai > a.mean()+a.std() else False for ai in a])
        else:
            b = np.copy(a)
        return b

    def bipolar(x):
        q = len(set(x))
        y = np.exp(1j * 2 * np.pi / q * x)
        return y

    zx = booleanize(x[1])
    zx = bipolar(zx)
    zx = (x[0],zx)

    zy = booleanize(y[1])
    zy = bipolar(zy)
    zy = (y[0],zy)

    return myxcor(zx,zy)

if __name__ == '__main__':
    x = np.random.binomial(1,0.1,1000)
    i = 2*np.array(range(len(x)))
    y = x
    x = (i,x)
    y = (i+77,y)
    r = logical_xcor(x,y)
    imax = np.argmax(r[1])
    print(r[0][imax], r[1][imax])


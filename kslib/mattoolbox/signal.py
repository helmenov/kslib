"""
Transported function from matlab signal processing toolbox    
"""
from scipy import signal as scisig
import numpy as np

def prepad(x,l:int,c=0):
    """prepad(x,l,c): xの長さがlになるまでcをxの先頭に追加する 
        transpoted from matlab.prepad
    Args:
        x (array): 元となる配列
        l (int): 出力される配列の長さ
        c (scalar): paddingされる値
    Output:
        (array) : 長さlにpaddingされた配列
    """
    x = np.array(x)
    lx = len(x)
    ret = np.full((l,),np.nan)
    ret[:l-lx] = c
    ret[l-lx:] = x 
    return ret

def postpad(x,l,c=0):
    """postpad(x,l,c): xの長さがlになるまでcをxの最後に追加する 
        transpoted from matlab.postpad
    Args:
        x (array): 元となる配列
        l (int): 出力される配列の長さ
        c (scalar): paddingされる値
    Output:
        (array) : 長さlにpaddingされた配列
    """
    x = np.array(x)
    lx = len(x)
    ret = np.full((l,),np.nan)
    ret[:lx] = x
    ret[lx:] = c 
    return ret

def resample(x,p,q):
    """resamples the input sequence, x, at p/q times the original sample rate. 
    `resample` applies an FIR Antialiasing Lowpass Filter to x 
    and compensates for the delay introduced by the filter. 
    The function operates along the first array dimension with size greater than 1.
    """
    if any({p,q}) <= 0:
        raise ValueError("p and q must be positive")
    while any({p,q}) < 1:
        p *= 10
        q *= 10
    
    great_common_divisor = np.gcd(p,q)
    if great_common_divisor > 1:
        p /= great_common_divisor
        q /= great_common_divisor
    #print(f'p:{p}, q:{q}')

    # properties of the anti-aliasing filter
    log10_rejection = -3.0
    stopband_cutoff_f = 1 / (2 * np.amax([p,q]))
    roll_off_width = stopband_cutoff_f / 10.

    # determine filter chunk_length
    # use empirical formula from [2] chap7, Eq.(7.63), p.476
    rejection_dB = -20*log10_rejection
    L = int(np.ceil((rejection_dB - 8.0) / (28.714 * roll_off_width)))

    # ideal sinc filter
    t = np.arange(-L,+L)
    ideal_filter = 2*p*stopband_cutoff_f*np.sinc(2*stopband_cutoff_f*t)

    # determine parameter of Kaiser window
    # use empirical formula from [2] Chap 7, Eq.(7.62), p.474
    if rejection_dB >= 21 and rejection_dB <= 50:
        beta = 0.5842 * (rejection_dB -21)**0.4 + 0.07886 * (rejection_dB -21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB -8.7)
    else:
        beta = 0.
    
    # apodize ideal filter response
    h = np.kaiser(2*L,beta) * ideal_filter

    Lx = len(x)
    Lh = len(h)
    L =  (Lh-1)/2
    Ly = int(np.ceil(Lx * p / q))

    # pre and posted filter response
    nz_pre = int(np.floor(q-np.mod(L,q)))
    hpad = prepad(h, Lh+nz_pre)

    offset = int(np.floor((L + nz_pre)/q))
    nz_post = 0
    while int(np.ceil(((Lx-1)*p + nz_pre + Lh + nz_post)/q)) - offset < Ly:
        nz_post += 1
    
    hpad = postpad(hpad, Lh + nz_pre + nz_post)

    # filtering
    xfilt = scisig.upfirdn(hpad, x, p, q)
    y = xfilt[offset+1:offset+Ly+1]
    return y

def scaling(f,xlim,ylim):
    xmu = 1/2*(xlim[0]+xlim[1])
    xd  = 1/2*(xlim[1]-xlim[0])
    ymu = 1/2*(ylim[0]+ylim[1])
    yd  = 1/2*(ylim[1]-ylim[0])
    f2 = (f-xmu)/xd*yd+ymu
    return f2

def upsampling(x,p,q):
    xu = []
    for k in range(len(x)-1):
        for i in range(p):
            xu.append((p-i)/p * x[k] + i/p *x[k+1])
            #print(f'\r[{k*p+i:7d}/{len(x)*p:7d}',end='')
    xud = xu[::q]
    return xud
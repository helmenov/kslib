import numpy as np
from scipy import signal as scisig
from kslib.mattoolbox import signal as matsig
from typing import List
from kslib import interpolate as kinp

def FireEngineSirenF0(t:List[float],num_harmonics:int=9):
    tt = t%6
    raise_region = np.where(np.logical_and(0<=tt,tt<2))[0]
    on_region = np.where(np.logical_and(2<=tt,tt<4))[0]
    fall_region = np.where(np.logical_and(4<=tt,tt<6))[0]

    f0 = np.full((len(t),),np.nan)
    f0[raise_region] = 390 + 250*np.log2(tt[raise_region]+1)
    f0[on_region] = 780
    f0[fall_region] = 780 - 170*(tt[fall_region]-4)
    return np.array([(h+1)*f0 for h in range(num_harmonics)]).T

def AmburanceSirenF0(t,num_harmonics=9):
    tt = t % 1.3
    on_region = np.where(np.logical_and(0<=tt,tt<0.65))[0]
    off_region = np.where(np.logical_and(0.65<=tt,tt<1.3))[0]
    f0 = np.full((len(t),),np.nan)
    f0[on_region] = 960
    f0[off_region] = 780
    return np.array([(h+1)*f0 for h in range(num_harmonics)]).T

def doppler_effect2(fs, duration, ve=1e2/9, vr=4/3, L=np.nan, D=20, vc = 340):
    """function applies changes in frequency that occurrs due to the doppler effect

        ```
        |---------L----------|
        (e)=======0==========
        →         | :
        ve        | D
                  | :
             vr ↑(r)-
                  |
        ```

    Args:
        fs : sampling frequency [Hz]
        duration : coef duration [s] 発音者の発音時間
        ve = 1e2/9 # in [m/s] 発音車の速度 default 1e2/9[=40*1000/(60*60)]<=40km/h
        vo = 4/3 # in [m/s] 受音者の速度 default 4/3[=80/60]<=80m/min
        L = np.nan # in [m] 発音車の総走行距離 default NaN
        D = 20 # in [m] 受音者の原点までの初期距離 default NaN
        vc = 340 # sound velocity [m/s]
    Return:
        fcoefs: # ドップラーシフト係数 いわゆるchirp rates
        gcoefs: # 距離減衰係数
        t: timestamp [s]
    """
    N = duration*fs
    t = np.arange(N)/fs

    # Lを設定していないときは，信号長の真ん中で目の前を通過するようにLを決める
    if np.isnan(L):
        L = ve*duration/2

    # 発音車のx座標
    xe = -L+ve*t
    # 受音者のy座標
    yr = -D+vr*t

    ve_cos = -xe/np.sqrt(xe**2 + yr**2)
    vr_cos = -yr/np.sqrt(xe**2 + yr**2)

    ve_on = ve * ve_cos
    vr_on = -vr * vr_cos

    fcoef = (vc-vr_on)/(vc-ve_on)
    gcoef = 1/(xe**2 + yr**2)

    return fcoef, gcoef, t

def doppler_effect(Xe,Xr):
    """Chirp rates by Doppler effect

    Args:
        Xe (_type_): List of (te, xe, ye, ze)
        Xr (_type_): List of (tr, xr, yr, zr)
    """
    te, *xe = Xe
    xe = np.array(xe)
    tr, *xr = Xr
    xr = np.array(xr)

    # Ve,vr
    def diff(x,t):
        vh = np.diff(x)/np.diff(t)
        th = (t[:-1]+t[1:])/2
        v = list()
        for vhi in vh:
            vi = kinp.lagrange((th,vhi),t)
            v.append(vi)
        v = np.array(v)
        return v
    ve = diff(xe,te)
    vr = diff(xr,tr)

    # Vc
    dt = (tr-te).clip(1e-7)
    vc = (xr-xe)/dt

    def product(A,B):
        return np.array([np.inner(a,b) for a,b in zip(A.T,B.T)])

    ve2 = np.array([vei[0]**2+vei[1]**2+vei[2]**2 for vei in ve.T])
    vr2 = np.array([vri[0]**2+vri[1]**2+vri[2]**2 for vri in vr.T])

    ret1 = (1-product(vc,vr))/(1-product(vc,ve))
    den = (1-vr2).clip(1e-7)
    ret2 = (1-ve2)/den
    ret2_sign = np.sign(ret2)
    ret2_sign = np.array([1 if i>=0 else -1 for i in ret2_sign])
    ret2_abs = np.abs(ret2)
    ret2 = ret2_sign * np.sqrt(ret2_abs)

    return ret1*ret2




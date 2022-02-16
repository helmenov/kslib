import numpy as np 
from scipy import signal as scisig
from kslib.mattoolbox import signal as matsig
from kslib import reduct_frac

def FireEngineSirenF0(t,num_harmonics=9):
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


def doppler_effect_from_signal(input, fs=44100, vs=40, vo=1/8, L=np.nan, D=20, precision=3, lenframe=64, loc_gain=False):
    """function applies changes in frequency that occurrs due to the doppler effect
    Args:
        input: 
        vs = 40 # in [km/h]
        vo = 1/8 # in [km/m]
        L = 100 # in [m] 
        D = 20 # in [m]
        precision = 3 # doppler shift fcoeffの精度パラメータ
                      #  (アップサンプリング 1/int(fcoeff):precision=0, (10^precision)/int(fcoeff*10^precision)
        lenframe = 64 # frame length 
    Return: 
        output: # ドップラー効果のかかった信号（長さはinputと異なることもある）
        fcoeff: # ドップラーシフト係数
    """
        
    nhop = lenframe

    N = len(input)
    duration = N/fs

    fcoef, gcoef, t = doppler_effect(fs,duration,vs,vo,L,D)

    if loc_gain == True:
        gain = gcoef
    else:
        gain = np.ones_like(t)

    average_fcoeff = np.zeros_like(fcoef)
    average_gain = np.zeros_like(gain)

    for istart in range(0,N,nhop):
        iend = np.amin([istart+lenframe,N])
        len_i = iend-istart
        avf = np.mean(fcoef[istart:iend])
        average_fcoeff[istart:iend] = int(np.mean(fcoef[istart:iend])*(10**precision))/(10**precision)
        avg = np.mean(gain[istart:iend])
        average_gain[istart:iend] = avg

        inp = input[istart:iend]
        outp = avg * matsig.resample(inp, int(10**precision), int(avf*(10**precision)))

        if istart == 0: 
            output = outp
            ystart = istart
        else:
            output = np.concatenate([output, outp])
        len_o = len(outp)
        ystart += len_o

    return output, fcoef, average_fcoeff, gain, average_gain

def doppler_effect_from_F0(f0, gharmonics, fs=44100,vs=40, vo=1/8, L=np.nan, D=20, precision=3, lenframe=64, loc_gain=False):
    """function applies changes in frequency that occurrs due to the doppler effect
    Args:
        f0 (duration(t-index), nchannel): 基本周波数
        vs = 40 # in [km/h]
        vo = 1/8 # in [km/m]
        L = 100 # in [m] 
        D = 20 # in [m]
        precision = 3 # doppler shift fcoeffの精度パラメータ
                      #  (アップサンプリング 1/int(fcoeff):precision=0, (10^precision)/int(fcoeff*10^precision)
        lenframe = 64 # frame length 
    Return: 
        output: # ドップラー効果のかかった信号（長さはinputと異なることもある）
        fcoeff: # ドップラーシフト係数
    """
    N = len(f0)
    duration = N/fs

    fcoef, gcoef, t = doppler_effect(fs, duration, vs, vo, L, D)

    nchannels = f0.shape[1]

    if loc_gain == True:
        gain = gcoef
    else:
        gain = np.ones_like(t)

    output = np.full((N,),0.0)
    for ichannel in range(nchannels):
        phase_doppler = 2*np.pi*np.cumsum(fcoef*f0[:,ichannel])/fs # int f0(t)*fcoeff(t) dt
        output +=  gharmonics[ichannel] * np.sin(phase_doppler)
    output *= gain

    return output, fcoef, gain

def doppler_effect(fs, duration, vs=40, vo=1/8, L=np.nan, D=20, vc = 340):
    """function applies changes in frequency that occurrs due to the doppler effect
    Args:
        fs : sampling frequency [Hz]
        duration : coef duration [s]
        vs = 40 # in [km/h]
        vo = 1/8 # in [km/m]
        L = 100 # in [m] 
        D = 20 # in [m]
        vc = 340 # sound velocity [m/s]
    Return: 
        fcoef: # ドップラーシフト係数
        gcoef: # 距離減衰係数
        t: timestamp [s]
    """
    N = duration*fs
    t = np.arange(N)/fs

    vs = vs*1000/(60*60)
    vo = vo*1000/(60)

    if np.isnan(L): # Lを設定していないときは，信号長の真ん中で目の前を通過するようにLを決める
        L = vs*N/fs/2 

    xs = -L+vs*t
    yo = -D+vo*t 

    vs_cos = -xs/np.sqrt(xs**2 + yo**2)
    vo_cos = -yo/np.sqrt(xs**2 + yo**2)

    vs_on = vs * vs_cos 
    vo_on = -vo * vo_cos 

    fcoef = (vc-vo_on)/(vc-vs_on)
    gcoef = 1/(xs**2 + yo**2)

    return fcoef, gcoef, t

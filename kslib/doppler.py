import numpy as np 
from scipy import signal as scisig
from mattoolbox import signal as matsig
import reduct_frac

def FireEngineSirenF0(t,harmonics=[2,1,1,1,1,1,1,1,1]):
    tt = t%6
    raise_region = np.where(np.logical_and(0<=tt,tt<2))[0]
    on_region = np.where(np.logical_and(2<=tt,tt<4))[0]
    fall_region = np.where(np.logical_and(4<=tt,tt<6))[0]

    f0 = np.full((len(t),),np.nan)
    f0[raise_region] = 390 + 250*np.log2(tt[raise_region]+1)
    f0[on_region] = 780
    f0[fall_region] = 780 - 170*(tt[fall_region]-4)
    return np.array([harmonics[h]*(h+1)*f0 for h in range(len(harmonics))]).T 

def AmburanceSirenF0(t,harmonics=[2,1,1,1,1,1,1,1,1]):
    tt = t % 1.3
    on_region = np.where(np.logical_and(0<=tt,tt<0.65))[0]
    off_region = np.where(np.logical_and(0.65<=tt,tt<1.3))[0]
    f0 = np.full((len(t),),np.nan)
    f0[on_region] = 960
    f0[off_region] = 780
    return np.array([harmonics[h]*(h+1)*f0 for h in range(len(harmonics))]).T 


def doppler_effect_from_signal(input, fs=44100, vs=40, vo=1/8, L=np.nan, D=20, precision=3, lenframe=64):
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
        
    vc = 340  # sound speed in [m/s]
    nhop = lenframe

    N = len(input)
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

    fcoeff = (vc-vo_on)/(vc-vs_on)
    gain = 1/(xs**2 + yo**2)


    average_fcoeff = np.zeros_like(fcoeff)
    average_gain = np.zeros_like(gain)

    for istart in range(0,N,nhop):
        iend = np.amin([istart+lenframe,N])
        len_i = iend-istart
        avf = np.mean(fcoeff[istart:iend])
        average_fcoeff[istart:iend] = int(np.mean(fcoeff[istart:iend])*(10**precision))/(10**precision)
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

    return output, fcoeff, average_fcoeff, gain, average_gain

def doppler_effect_from_F0(f0, fs=44100, vs=40, vo=1/8, L=np.nan, D=20, precision=3, lenframe=64, loc_gain=False):
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
        
    vc = 340  # sound speed in [m/s]

    N = len(f0)
    nchannels = np.ndim(f0)
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

    fcoeff = (vc-vo_on)/(vc-vs_on)
    if loc_gain == True:
        gain = 1/(xs**2 + yo**2)
    else:
        gain = np.ones_like(t)

    output = np.full((N,),0.0)
    for ichannel in range(nchannels):
        gharmonics = 2**(-ichannel)
        phase_doppler = 2*np.pi*np.cumsum(fcoeff*f0[:,ichannel])/fs # int f0(t)*fcoeff(t) dt
        output +=  gharmonics * np.sin(phase_doppler)
    output *= gain

    return output, fcoeff, gain
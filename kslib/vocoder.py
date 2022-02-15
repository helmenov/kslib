import numpy as np
from scipy import signal as scisig
from kslib import reduct_frac

def formant_filter(x,fs,gender,vowel):
    y = np.copy(x)

    #                 /i/,  /e/,  /a/,  /o/,  /u/    
    formant        = np.array([[[ 230,  450,  750,  525,  350],   # F1 male
                                [2150, 1950, 1300,  975, 1450],   # F2
                                [2850, 2700, 2600, 2600, 2400]],  # F3
                               [[ 275,  575,  950,  650,  350],   # F1 female
                                [2800, 2200, 1550, 1250, 1650],   # F2
                                [3650, 3050, 2900, 2850, 2750]]]) # F3
    formant_gain_dB = np.array([[   8,   10,   11,   12,    9],   # F1
                                 [-12,   -5,    7,    5,   -7],   # F2
                                 [-16,  -12,  -16,  -22,  -31]],  # F3
                                 dtype='float64')
    formant_gain = 10**(formant_gain_dB/20.0)  # log->linear
    formant_bandwidth = np.array([50,  # F1
                                  64,  # F2
                                 115], # F3
                                 dtype='float64')

    gender_set = {"male":0, "female":1}
    vowel_set = {"i":0, "e":1, "a":2, "o":3, "u":4}

    s = gender_set[gender]
    v = vowel_set[vowel]

    # フォルマント周波数の共振フィルタをかける
    nformants = formant.shape[1]
    alpha = formant_bandwidth * np.pi /fs
    y2 = np.zeros_like(y)
    for k in range(nformants):
        b = np.full((3,),0.0)
        a = np.full((3,),0.0)
        a[0] = 1.0
        w  = 2.0*np.pi*formant[s,k,v]/fs
        b[1] = ((alpha[k]**2 + w**2)/w) * np.sin(w) * np.exp(-alpha[k])
        b = b * formant_gain[k,v]
        a[1] = -2*np.exp(-alpha[k])*np.cos(w)
        a[2] = np.exp(-2.0*alpha[k])
        y2 += scisig.lfilter(b,a,y)
    return y2
    
def vocal(F0,duration,fs,beta=0):
    SEED = 83988848
    rg = np.random.default_rng(seed=SEED)
    """
    voice : impulse sequence
    voiceless : white noise
    source = (1-beta) * voiced + beta * voiceless
    """
    t = np.arange(duration*fs)/fs
    lt = len(t)
    # 声帯振動を作成
    """
    p,q = reduct_frac.reduct_frac(fs,F0)  # 基本周期
    while np.amin([np.log10(p),np.log10(q)]) > np.log10(lt):
        p = (lambda x: int((x * 2 + 1) // 2))(p/10)
        q = (lambda x: int((x * 2 + 1) // 2))(q/10)
    """
    p, q = (lambda x: int((x * 2 + 1) // 2))(fs/F0), 1
    #print(f'{fs}/{F0}={fs/F0} is reducted to {p}/{q}={p/q}')
    
    if lt*q < 2*p:
        raise ValueError(f'you should set duration loger than {np.ceil(2*p/q)}')
    voice_ = np.full((lt*q,),0.0)
    voice_[0::p] = 0.5
    voice = np.full((lt,),0.0)
    for i in range(lt):
        voice[i] = np.amax(voice_[q*i:q*(i+1)])
    source = (1-beta)*voice + beta*0.5*rg.standard_normal(size=(lt,))
    return source

def f0_typical(chroma,key):
    chroma_dic = {
        "A": 0,
        "Ais":1,
        "B":2,
        "C":3,
        "Cis":4,
        "D":5,
        "Dis":6,
        "E":7,
        "F":8,
        "Fis":9,
        "G":10,
        "Gis":11,
    }

    F0 = (chroma_dic[chroma]/12+1)*440 * 2**(key-4)

    return F0

import numpy as np

def from_harmonics(f0,harmonics,duration=5,fs=44100,sin_or_cos='sin'):
    """
    sin使用と，cos使用
    - 時間波形は異なる．sinでsawtoothでもcosだとそうならない
    - 振幅スペクトルは同じ
    - 位相スペクトルは異なる
    - 人間の耳では同じ
    """
    
    t = np.arange(duration*fs)/fs
    components = lambda i,t: harmonics[i] * np.sin(2*np.pi*f0*(i+1)*t)
    
    if harmonics[2*i] == 1/(2*i+1):
        if harmonics[2*i+1] == 1/(2*(i+1)):
            print("sawtooth")
        else:
            print("rectangular")

def gen_delay_reverb(fs, atten=0.5, delay=0.05, reverb=[1,0.8,0,1,0.8], repeat=10):
    """

    y(n) = sum_m b(m) * x(n-m)
    """
    lr = len(reverb)
    lb = (repeat+1)*np.max([delay*fs,lr])
    b = np.full((lb,),0)
    b[1] = 1
    for r in range(repeat):
        b[r*delay:r*delay+lr] = atten**(r+1) * reverb

    return b


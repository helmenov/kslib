import numpy as np
from scipy import fftpack as scifft
from typing import Tuple

def freq2mel(freqs, f0:float=700):
    '''convert frequency [Hz] to mel-frequency [mel]
    .. math:: usepackage{physics} m = m_0 log_e qty(frac{f}{f_0}+1)
    and m_0 is fixed by 1000[Hz] converted 1000[mel]

    Args:
        freq : frequency [Hz]
        f0 (float, optional): . Defaults to 700.
    Returns:
        : mel-frequency [mel]
    '''
    m0 = 1000/np.log(1000/f0+1)
    mels = m0 * np.log(freqs/f0+1)
    return mels

def mel2freq(mels, f0:float=700):
    m0 = freq2mel(1000,f0)
    freqs= f0 * (np.exp(mels/m0) - 1)
    return freqs

def oct2ratio(octs):
    ratios = 2**octs
    return ratios

def ratio2oct(ratios):
    octs = np.log2(ratios)
    return octs

def cent2oct(cents):
    octs = cents/1200
    return octs

def oct2cent(octs):
    cents = 1200*octs
    return cents

def Qfactor(fc,fl,fh):
    Q = fc/(fh-fl)
    return Q

def BandN(fmin,fmax,Nband):
    """(fmin,fmax)にNband個の対数周波数バンドパスフィルタを作る

    _l : log-scaled

    Args:
        fmin (_type_): _description_
        fmax (_type_): _description_
        Nband (_type_): _description_

    Returns:
        np.array([fL, fC, fH])
        Q
        oct
    """
    EPS = 1e-7
    fmax_l = np.log(fmax)
    fmin_l = np.log(fmin+EPS)
    d_l = (fmax_l-fmin_l) / (Nband+1)
    fL_l = fmin_l + np.arange(0,Nband)*d_l
    fC_l = fL_l + d_l
    fH_l = fC_l + d_l
    fC = np.exp(fC_l)
    fL = np.exp(fL_l)
    fH = np.exp(fH_l)
    Q = Qfactor(fC[0],fL[0],fH[0])
    oct = ratio2oct(fH[0]/fL[0])
    return np.array([fL,fC,fH]), Q, oct

def BandFc(fc,param=1,mode="oct"):
    """band

    Args:
        fc (_type_): band center frequency
        mode (str, optional): "oct", "Q", "cent". Defaults to "oct".
    Returns:
        : [f_L, f_H], lower and higher bound of band-pass or filter bank
    """
    # log(fc) := 1/2 * (log(fL) + log(fH))
    if mode=="oct":
        oct = param
        # f_H := f_L * 2^{oct}
        fL = 2 ** (-oct/2) * fc
        fH = fL * 2**oct
    elif mode=="Q":
        q = param
        # Q := fc/(fH-fL)
        # => fc = q(fH-fL)
        # => fH = fc/q + fL
        # fc = np.sqrt(fL * (fc/q + fL))
        # fc^2 = fc/q * fL + fL^2
        # fL^2 + fc/q fL - fc^2 = 0
        a = 1
        b = fc/q
        c = -fc**2
        fL = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        fH = fc/q + fL
    elif mode=="cent":
        cent = param
        # fH = fL*2^{cent/1200}
        # fc^2 = fL^2 * 2^{cent/1200}
        fL = 2**(-cent/1200/2) * fc
        fH = fL * 2**(cent/1200)
    return np.array([fL, fH])




def filterbank(N,fs,f0=700,fcs=None):
    f_nyq = fs/2
    N_nyq = N/2
    df = f_nyq/N_nyq
    if fcs is None:
        Nopt = np.ceil(fs/f0/(2**(1/1200)-1)) # 1cent幅 > fs/N
        df = fs/Nopt
        Nband = int(np.floor(1/np.log2(df/f0/2+1)))
        print(Nband)
        fLCH, Q, oct = BandN(fmin=0, fmax=f_nyq, Nband=Nband)
        print(fLCH.T)
        H, fLCH = filterbank(N=N, fs=fs, fcs=fLCH[1])
    else:
        fcs = np.array(fcs)
        N_bank = len(fcs) if fcs.ndim else 1
        if fcs[-1] > f_nyq:
            raise ValueError(f'f_c must be lower than {f_nyq}')

        fL = np.r_[0, fcs[:-1]]
        fH = np.r_[fcs[1:], f_nyq]
        iC = np.array(fcs/df)
        iL = np.array(fL/df)
        iH = np.array(fH/df)

        H = np.zeros((N_bank,N))
        for n in range(N_bank):
            #fCgain = 1/(fH[n]-fL[n])
            #H[n,:int(iC[n])] = 1/2*( (np.arange(int(iC[n]))-iL[n]) / (iC[n]-iL[n]) + 1)
            #H[n,int(iC[n]):int(N_nyq)] = 1/2*( (np.arange(int(iC[n]),int(N_nyq))-int(iH[n]))/ (iC[n]-iH[n]) +1)

            #fCgain = 2/(fH[n]-fL[n])
            fCgain = 1
            H[n,int(iL[n]):int(iC[n])] = np.arange(int(iC[n])-int(iL[n]))/(iC[n]-iL[n])
            H[n,int(iC[n]):int(iH[n])] = 1-np.arange(int(iH[n])-int(iC[n]))/(iH[n]-iC[n])

            H[n,-1:-int(np.ceil(N_nyq)):-1] = H[n,1:int(np.ceil(N_nyq))]
            H[n] *= fCgain
        fLCH = np.array([fL, fcs, fH])

    return H, fLCH

def rcunwrap(x: np.ndarray) -> Tuple[np.ndarray, int]:
    '''
        unwrap  the phase and removes phase corresponding to integer lag.
    '''
    n: int = len(x)
    nh: int = int(np.floor((n+1)/2))
    y: np.ndarray = np.unwrap(x)
    nd: int = np.round((y[nh]/np.pi))
    for i in range(len(y)):
        y[i] = y[i] - np.pi*nd*i/nh
    return y, nd

class fourierAnalysis():
    def __init__(self, x, Fs):
        self.Fs = Fs
        f_nyq = Fs/2
        N = len(x)
        N_nyq = N/2
        freqs = np.zeros_like(x)
        freqs[0:int(np.ceil(N_nyq))] = np.arange(N_nyq)*f_nyq/N_nyq
        freqs[-1:-int(np.ceil(N_nyq)):-1] = -freqs[1:int(np.ceil(N_nyq))]
        self.freqs = np.array(freqs)
        X = scifft.fft(x)
        #X = X[:int(np.floor(len(X)/2+1))]

        self.__X = X

        A = np.abs(X)
        self.__amp = A

        logA = np.log(np.array([a.clip(min=1e-7) for a in A]))

        ## weighted logA
        L = len(logA)
        w_logA = np.hanning(L+1)[:L] * logA

        self.__dB = 20 * logA

        ###
        #  Phi : phase spectrum
        Phi,ndelay = rcunwrap(np.angle(X))
        self.__phase = Phi
        self.ndelay = ndelay

        C_amp = scifft.ifft(logA)
        C_amp = C_amp[:int(np.floor(len(C_amp)/2+1))]

        self.__C_amp = C_amp

        jPhi = np.array([complex(0,p) for p in Phi])

        C_phase = scifft.ifft(jPhi)
        C_phase = C_phase[:int(np.floor(len(C_phase)/2+1))]

        self.__C_phase = C_phase

        lenC = len(C_amp)
        # integer index of N/2, that is the indices from next index are 'negative'.
        nyq = int(np.floor((lenC-1)/2))

        C_min = np.zeros_like(C_amp)
        C_min[0] = C_amp[0]
        C_min[1:int(nyq+1)] = 2*np.ones((nyq,))*C_amp[1:int(nyq+1)]
        self.__C_min = C_min

        C_all = C_amp+C_phase-C_min
        self.__C_all = C_all

        C_phase_all = C_all
        self.__C_phase_all = C_phase_all

        C_phase_min = C_min - C_amp
        self.__C_phase_min = C_phase_min

        C_amp_min = C_min - C_phase_min
        self.__C_amp_min = C_amp_min

        C_amp_all = C_amp - C_amp_min
        self.__C_amp_all = C_amp_all

        Phi_min = np.unwrap(np.imag(scifft.fft(C_phase_min)))
        self.__phase_min = Phi_min

        Phi_all = np.unwrap(np.imag(scifft.fft(C_phase_all)))
        self.__phase_all = Phi_all

        A_min = np.exp(np.real(scifft.fft(C_amp_min)))
        self.__amp_min = A_min

        A_all = np.exp(np.real(scifft.fft(C_amp_all)))
        self.__amp_all = A_all

        jPhi_min = np.array([complex(0,p) for p in Phi_min])
        X_min = A_min * np.exp(jPhi_min)
        self.__X_min = X_min

        jPhi_all = np.array([complex(0,p) for p in Phi_all])
        X_all = A_all * np.exp(jPhi_all)
        self.__X_all = X_all

        xmin = np.real(scifft.ifft(X_min))
        self.__xmin = xmin

        xall = np.real(scifft.ifft(X_all))
        self.__xall = xall

    @property
    def X(self):
        """Complex Spectrum
        """
        return self.__X

    @property
    def amp(self):
        """Amplitude Spectrum (real)
        """
        return np.real(self.__amp)

    @property
    def dB(self):
        """Power Spectrum in dB (real)
        """
        return np.real(self.__dB)

    @property
    def phase(self):
        """Phase Spectrum (real)
        """
        return np.real(self.__phase)

    @property
    def rcphase(self):
        """Phase Spectrum -RC (real)
        """
        return np.real(self.__rcphase)

    @property
    def C_complex(self):
        """Complex Cepstrum (complex)
        """
        #assert np.all((self.__C_min + self.__C_all) == (self.__C_amp + self.__C_phase))
        return self.__C_min + self.__C_all

    @property
    def C_amp(self):
        """Amplitude Cepstrum
        Even func.
        """
        return self.__C_amp

    @property
    def C_phase(self):
        """Phase Cepstrum
        Odd func.
        """
        return self.__C_phase

    @property
    def C_min(self):
        """Complex Cepstrum with Minimum Phase
        """
        return self.__C_min

    @property
    def C_all(self):
        """Complex Cepstrum with Allpass
        """
        return self.__C_all

    @property
    def C_phase_all(self):
        """Phase Cepstrum with Allpass
        Odd func.
        """
        return self.__C_phase_all

    @property
    def C_phase_min(self):
        """Phase Cepstrum with Minimum Phase
        Odd func.
        """
        return self.__C_phase_min

    @property
    def C_amp_min(self):
        """Amplitude Cepstrum with Minimum Phase
        Even func.
        """
        return self.__C_amp_min

    @property
    def C_amp_all(self):
        """Amplitude Cepstrum with Allpass
        Even func.
        """
        return self.__C_amp_all

    @property
    def phase_min(self):
        """Phase Spectrum with Minimum Phase
        """
        return self.__phase_min

    @property
    def phase_all(self):
        """Phase Spectrum with Allpass
        """
        return self.__phase_all

    @property
    def amp_min(self):
        """Amplitude Spectrum with Minimum Phase
        """
        return self.__amp_min

    @property
    def amp_all(self):
        """Amplitude Spectrum with Allpass
        """
        return self.__amp_all

    @property
    def X_min(self):
        """Complex Spectrum with Minimum Phase
        """
        return self.__X_min

    @property
    def X_all(self):
        """Complex Spectrum with Allpass
        """
        return self.__X_all

    @property
    def xmin(self):
        """Minimum Phase Component
        """
        return self.__xmin

    @property
    def xall(self):
        """Allpass Component
        """
        return self.__xall

    def mel(self,Nband,f0=700):
        """mel spectrogram
        """
        assert Nband > 0
        N = len(self.X)
        f_nyq = self.Fs/2
        m_nyq = freq2mel(freqs=f_nyq,f0=f0)
        m_c = np.linspace(0,m_nyq,Nband+2)[1:-1]
        f_c = mel2freq(mels=m_c,f0=f0)
        H = filterbank(N=N,fs=self.Fs,fcs=f_c)
        return H @ self.X, f_c

    def logf(self):
        N = len(self.X)
        H, fLCH = filterbank(N=N,fs=self.Fs)
        return H @ self.X, H, fLCH[1]


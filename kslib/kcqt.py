import numpy as np
from scipy import fftpack as scifft
from scipy import sparse as scisparse

def cqt_kernel(fs,fmin,fmax,q,mode="Q"):
    """
    * 周波数成分に応じてサンプル長を変えてDFTする．
    * 周波数は，fk = fmin * 2^{k/b} k=0,...,b, b+1, ..., K-1 を選ぶ．
      K <- arg[ fmax = fmin * 2^{(K-1)/b} ]
      * つまり，最初に fminとfmaxとbを決めると K が決まる
    * この周波数選択フィルターは，[fk, fk*2^{1/b}]の選択フィルターなので，
    .. math :: Q = fk / (fk*2^{1/b} - fk) = 1/ (2^{1/b} - 1)
      も満たす．(Qが求まる．(Qを先に与えるとbが決まる)
    * このQは，サンプリング定理より
    .. math :: Q = fk / (fs/N[k] - eps)
      なので，fs/ N[k] = fk / Q つまり N[k] = fs/fk * Q
    * DFTの式は，
    .. math :: X[fk] = 1/N[k] \sum x[n] * exp(-j 2pi k n/N[k])
    * k fs/N[k] = fk だから，k = fk/fs*N[k]
    * k = fk/fs*N[k] = fk/fs*fs/fk*Q = Q
    .. math :: X[fk] = 1/N[k] \sum x[n] * exp(-j 2pi Q n/N[k])

    * 行列演算用に直すと，
    X_{cqt} = x @ conj(T)
         T_{nk} = 1/N[k] exp(j 2pi Q n/N[k]) for n < N[k]
                = 0                          for otherwise
    * Perseval equality より
         <x,conj(y)> = 1/N <X, conj(Y)>, X=F(x), Y=F[y]
    * したがって，
    X_{cqt} = 1/N F(x) @ F(T)
      つまり，xのFourier変換を求めたあとに，カーネルを行列乗算すれば求まる
      F(T)はスパースになりやすいので，演算が早い．
    """

    if mode == "Q": # oct_devide b is calculated from Q
        b = np.round(1/np.log2(1/q + 1))
    else: # as "b", Q is calculated from oct_devide b
        b = int(q)
    Q = 1/(2**(1/b)-1) # Q is resolved as float

    # fmax-eps = fmin * 2^{K/b}
    K = np.round(b * np.log2(fmax/fmin) + 1).astype('int')

    def nextpow(x:float)->int:
        """xより大きい2冪整数

        Args:
            x (float): _description_

        Returns:
            int: _description_
        """
        return (2 ** np.ceil(np.log2(x))).astype('int')

    N = nextpow(fs/fmin*Q)

    S = np.zeros((N,K),dtype='complex')
    for k in range(K):
        Tk = np.zeros((N,),dtype='complex')
        fk = fmin * 2**(k/b)
        Nk = int(fs/fk*Q)
        win = np.hamming(Nk+1)[:-1]
        iBegin_Window = int((N - Nk)/2)
        Tk[iBegin_Window:iBegin_Window+Nk] = win / Nk * np.exp(1j * 2 * np.pi * Q * np.arange(Nk)/Nk)
        S[:,k] = scifft.fft(Tk)


    Th = 1e-2
    S[abs(S) <= Th] = 0
    S = scisparse.csr_matrix(S)

    return S


def cqt(x,fs,fmin=60, fmax=None):
    L = len(x)
    F_nyq = fs/2

    if fmax is None:
        fmax = F_nyq
    b = 24
    S = cqt_kernel(fs,fmin=fmin,fmax=fmax,q=b,mode='b')
    nfft = S.shape[0]
    K = S.shape[1]

    hnfft = int(nfft/2)
    xx = np.zeros(L+nfft)
    xx[hnfft:-hnfft] = x

    sec_hop = 0.01
    lhop = int(np.round(sec_hop*fs))
    nframe = int(L/lhop)

    X = np.zeros((nframe, K), dtype='complex')
    for iframe in range(nframe):
        ibegin = iframe*lhop
        iend = ibegin + nfft
        Xi = scifft.fft(xx[ibegin:iend])
        X[iframe] = Xi @ np.conj(S)
    X /= nfft

    fk = np.array([fmin * 2**(k/b) for k in np.arange(K)])
    tk = np.array([lhop*n/fs for n in np.arange(nframe)])

    return X, fk, tk







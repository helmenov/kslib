import numpy as np
from scipy import fftpack as scifft

def f_fit(ids, vals, method='Cubic'):
    ids = np.array(ids)
    if method == 'Lagrange':
        def closure(x,ids=ids,vals=vals):
            L = len(ids)
            lp = np.zeros((L,))
            for i, xi in enumerate(ids):
                lp[i] = 1
                for xj in np.delete(ids,i):
                    lp[i] *= (x-xj)/(xi-xj)

            f = 0
            for i in range(L):
                f += vals[i]*lp[i]
            return f
    elif method == 'Sinc':
        def closure(x,ids=ids,vals=vals):
            L = len(ids)
            lp = np.zeros((L,))
            for i, xi in enumerate(ids):
                lp[i] = 1
                for xj in np.delete(ids,i):
                    lp[i] *= (x-xj)/(xi-xj)
            f = 0
            for i in range(L):
                f += vals[i]*np.sinc(lp[i])
            return f
    elif method == 'Cubic':
        def closure(x,ids=ids,vals=vals):
            idx = np.hstack([(np.where(x-ids>0)[0])[:-3:-1],(np.where(x-ids<0)[0])[:2]])
            ids = ids[idx]
            vals = vals[idx]
            L = len(ids)
            lp = np.zeros((L,))
            for i, xi in enumerate(ids):
                lp[i] = 1
                for xj in np.delete(ids,i):
                    lp[i] *= (x-xj)/(xi-xj)

            f = 0
            for i in range(L):
                f += vals[i]*lp[i]
            return f

    return closure

def SD(post, pre):
    y = 0
    for t in range(len(pre)):
        if 1e-7 < np.abs(post[t]) < 1e+7:
            if y < 1e+150:
                y = y + ((post[t] - pre[t])**2) / (post[t]**2)
    return y

def check_IMF(x):
    """imf must satisfy following
    1. n_extrema and n_cross must be eual or differ at most by one
    2. maxima must be positve, and minima must be negative
    """
    i_min, i_max, n_extrema, i_cross, n_cross = extrema(x)
    numDiffExtremaCross = np.abs(n_extrema-n_cross)
    numNegativeMaxima = len(np.where(x[i_max]<0)[0])
    numPositiveMinima = len(np.where(x[i_min]>0)[0])
    if numDiffExtremaCross <= 1 and numNegativeMaxima == 0 and numPositiveMinima == 0:
        P = True
    else:
        P = False

    return P, numDiffExtremaCross, numNegativeMaxima, numPositiveMinima

def check_MeanIsAlwaysZero(x_mean):
    return x_mean.std() / (1e-7)

def imf(signal, noise_assisted = 100000, N_inner=1000, N_outer=1000):
    if noise_assisted == 0:
        N_ite = 1
    elif noise_assisted >0:
        N_ite = noise_assisted
    rg = np.random.default_rng()
    x = signal
    len_x = len(x)
    IMFs = list()
    n = 0
    while n < N_outer:
        h = x
        k = 0
        while k < N_inner:
            v_mean = np.zeros_like(h)
            sigma_h = np.std(h)
            for _ in range(N_ite):
                if noise_assisted == 0:
                    hn = h
                else:
                    Noise = rg.normal(loc=0, scale= 0.1*sigma_h, size=h.shape)
                    hn = h + Noise

                i_min, i_max, n_extrema, i_cross, n_cross = extrema(hn)

                if len(i_min) > 0 and len(i_max) > 0 and i_min[0] < i_max[0]:
                    i_max = np.hstack([0, i_max]).astype(int)
                else:
                    i_min = np.hstack([0, i_min]).astype(int)
                if len(i_min) > 0 and len(i_max) > 0 and i_min[-1] > i_max[-1]:
                    i_max = np.hstack([i_max,len_x-1]).astype(int)
                else:
                    i_min = np.hstack([i_min,len_x-1]).astype(int)

                f_min = f_fit(i_min, hn[i_min])
                f_max = f_fit(i_max, hn[i_max])
                indices = np.arange(len_x)
                v_mean = v_mean + (np.array(list(map(f_min,indices))) + np.array(list(map(f_max,indices))))/2
            v_mean /= N_ite

            h_new = h - v_mean

            P0 = check_MeanIsAlwaysZero(v_mean)
            P1, numDiffExtremaCross, numNegativeMaxima, numPositiveMinima = check_IMF(h_new)
            SD_inner = SD(h_new,h) # SD == 0 is equivalent with v_mean.std == 0
            print(f' | MeanStd:{P0:2.1g}, numDiffExtremaCross:{numDiffExtremaCross}, numCross:{n_cross}, numNegativeMaxima:{numNegativeMaxima}, numPositiveMinima:{numPositiveMinima}, SD_inner:{SD_inner:2.1g}')
            if P1 and SD_inner < 0.3:
                IMFs.append(h_new)
                break
            else:
                k = k+1

            h = h_new

        x_new = x - IMFs[-1]
        SD_outer = SD(x_new,x)
        assert ~np.isnan(SD_outer)
        print(f'SD_outer:{SD_outer}')
        if SD_outer < 0.3:
            res = x_new
            break
        else:
            n = n+1
        x = x_new
    res = x_new

    return np.array(IMFs), res

def HilbertHuang(x):
    IMFs, res = imf(x)
    n_mode = len(IMFs)
    InstantFreq = np.zeros((IMFs.shape[0],IMFs.shape[1]-1))
    InstantPow = np.zeros_like(IMFs)
    for i in range(n_mode):
        xi = IMFs[i]
        nyq = int(np.floor((len(xi)-1)/2))
        spectrum = scifft.fft(xi)
        spectrum[1:nyq] *= 2
        spectrum[nyq:] = 0
        analytic = scifft.ifft(spectrum)
        phases = np.angle(analytic)
        InstantFreq[i] = np.diff(phases)/(2*np.pi)
        InstantPow[i] = np.abs(analytic)
    return InstantFreq, InstantPow

def AMDF(x):
    """Average Magnitute Difference Function"""
    len_x = len(x)
    Range = np.arange(-(lenx-1),len_x)
    ret = list()
    for tau in Range:
        y = 0
        for t in range(-tau,len_x-tau):
            y = y + np.abs(x[t] - x[t+tau])
        ret.append(y)
    return np.array(ret)

def WAC(x, weight='AMDF'):
    """Weighted AutoCorrelation Function"""
    if weights=='AMDF':
        Weights = AMDF(x)
    elif weights == 'Energy':  # its named NACF:Normalize AutoCorrelation Function
        len_x = len(x)
        e0 = np.sqrt(np.sum(x**2))
        Range = np.arange(-(lenx-1),len_x)
        etau = list()
        for tau in Range:
            e = 0
            for t in range(-tau,len_x-tau):
                e = e + x(t+tau)**2
            etau.append(e)
        etau = np.sqrt(np.array(etau))
        Weights = e0*etau

    xcor = np.correlate(x,x,'full')
    ret = xcor/(Weights+1e-7)
    return ret




def extrema(x):
    """
    find indices of extrema and zero-cross

    Output:
        i_min: indices of local minima
        i_max: indices of local maxima
        n_extrema: num of extrema (that is len(i_min) + len(i_max))
        i_cross: float indices zero-cross
        n_cross: num of zero-cross
    """
    len_x = len(x)
    derivativeSignSlices = np.sign(np.diff(x))
    i_min = list()
    i_max = list()
    i_cross = list()
    n_min = 0
    n_max = 0
    for i in range(1,len_x-2):
        if derivativeSignSlices[i] == 0:
            if derivativeSignSlices[i-1] < 0 and derivativeSignSlices[i+1] > 0:
                i_min.append(i)
                i_min.append(i+1)
                n_min += 1
            elif derivativeSignSlices[i-1] > 0 and derivativeSignSlices[i+1] < 0:
                i_max.append(i)
                i_max.append(i+1)
                n_max += 1
        elif derivativeSignSlices[i] < 0:
            if derivativeSignSlices[i+1] >0:
                i_min.append(i+1)
                n_min += 1
        else: # derivativeSignSlices[i] > 0
            if derivativeSignSlices[i+1] <0:
                i_max.append(i+1)
                n_max += 1
    n_extrema = n_min + n_max

    Signs = np.sign(x)
    for i in range(1,len_x-1):
        if Signs[i] == 0:
            if Signs[i-1] * Signs[i+1] < 0:
                i_cross.append(i)
        elif Signs[i] * Signs[i+1] < 0:
            i_c = i + x[i]/(x[i]-x[i+1])
            i_cross.append(i_c)
    n_cross = len(i_cross)

    return i_min, i_max, n_extrema, i_cross, n_cross




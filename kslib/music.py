"""
音楽関係の変換
- note2freq(note,octave,A4=440) -> freq
- freq2note(freq,A4=440) -> (note,octave)
- cent2ratio(cent) -> ratio:Fraction
- ratio2cent(ratio:Fraction) -> cent
- octave2harmonic(octave) -> harmonic:Fraction
- harmonic2octave(harmonic:Fraction) -> octave
"""
from fractions import Fraction
from kslib.reduct_frac import reduct_frac as ksfrac
import numpy as np

def note2freq(note,octave,A4=440):
    """音名とオクターブから周波数に変換

    Args:
        note (_type_): 音名 in {A, B-, B, C, D-, D, E-, E, F, G-, G, A-}
        octave (_type_): オクターブ [oct]
        A4 (int, optional): A4の周波数 [Hz]. Defaults to 440.

    Returns:
        周波数: in [Hz]
    """
    l_note = ['A','B-','B','C','D-','D','E-','E','F','G-','G','A-']
    assert note in l_note
    l = l_note.index(note)
    if l == 0 and octave == 4:
        freq = A4
    else:
        freq = note2freq('A',4) * 2**(octave-4) * cent2ratio(100.0*l)
    return round(freq)

def freq2note(freq, A4=440):
    """周波数を音名とオクターブに変換

    Args:
        freq (_type_): 周波数 in [Hz]

    Returns:
        (note, octave): 音名，オクターブ
    """
    l_note = ['A','B-','B','C','D-','D','E-','E','F','G-','G','A-']
    j_min = 100000
    for cc in range(12):
        for oct in range(10):
            j = (4.0 - np.log2(A4) + np.log2(freq) - oct - cc/12.0)**2
            if j < j_min:
                j_min = j
                oct_ = oct
                cc_ = cc
    note = l_note[cc_]
    octave = oct_
    return note,octave

def cent2ratio(cent):
    """scale cent to ratio

    Args:
        cent (_type_): cent

    Returns:
        _type_: ratio

    Example:
    >>> cent2ratio(1200)
    Fraction(2, 1)
    """
    cent = np.array(cent)
    ratio = 2.0**(cent/1200.0)
    ratio = Fraction(*ksfrac(ratio,1))
    return ratio

def ratio2cent(ratio,denominator=1):
    """scale ratio to cent

    Args:
        ratio (_type_): ratio

    Returns:
        _type_: cent

    Example:
    >>> ratio2cent(2,1)
    1200.0
    """
    if denominator == 1:
        if isinstance(ratio, list) or isinstance(ratio, tuple):
            ratio = Fraction(*ratio)
        elif isinstance(ratio, Fraction):
            pass
        elif isinstance(ratio, int) or isinstance(ratio,float):
            ratio = Fraction(ratio,denominator)
        else:
            raise ValueError
    else:
        if isinstance(ratio, int) or isinstance(ratio, float):
            ratio = Fraction(ratio,denominator)
        else:
            raise ValueError

    cent = 1200*(np.log2(ratio.numerator)-np.log2(ratio.denominator))
    return cent

def octave2harmonic(octave):
    """ octave間隔値を周波数比に変換

    Args:
        octave (_type_): octave間隔値

    Returns:
        _type_: 周波数比

    Example:
    >>> octave2harmonic(1)
    Fraction(2, 1)

    """
    harmonic = 2.0 ** octave
    harmonic = Fraction(*ksfrac(harmonic,1))
    return harmonic

def harmonic2octave(harmonic,denominator=1):
    """周波数比をoctaveスケール値に変換

    Args:
        harmonic = (num,den): num/den 周波数比

    Returns:
        _type_: octaveスケール値

    Example:
    >>> harmonic = Fraction(2,1)
    >>> harmonic2octave(harmonic)
    1.0
    """
    if denominator == 1:
        if isinstance(harmonic, list) or isinstance(harmonic, tuple):
            harmonic = Fraction(*harmonic)
        elif isinstance(harmonic, Fraction):
            pass
        elif isinstance(harmonic, int) or isinstance(harmonic,float):
            harmonic = Fraction(harmonic,denominator)
        else:
            raise ValueError
    else:
        if isinstance(harmonic, int) or isinstance(harmonic, float):
            harmonic = Fraction(harmonic,denominator)
        else:
            raise ValueError
    octave = np.log2(harmonic.numerator) - np.log2(harmonic.denominator)
    return octave

def interval_correlate(x,y,ti=1e-2,standard=True):
    """interval_correlate:
    2つのtiming dataの相互相関

    Input:
        x,y : timing data
        ti : tick interval, defalut:1e-7
        standard: Standardize option, default:True
    Output:
        t : timing delay. same dimension with x,y
        xcor : cross correlations

    >>> x = [1, 2, 4, 7, 9] # f(t) = 1 if t in x else 0
    >>> y = [4, 7, 10, 12]  # g(t) = 1 if t in y else 0
    >>> t, xcor = interval_correlate(x,y)
    """
    xxl = np.ceil(np.max(x) / ti).astype(int)
    yyl = np.ceil(np.max(y) / ti).astype(int)
    N = xxl if xxl > yyl else yyl
    tl = xxl + yyl - 1
    t = (np.arange(tl)-yyl+1)*ti

    xx = np.zeros(xxl)
    xx[(x / ti).astype(int)] = 1
    xx = np.r_[np.zeros(yyl-1),xx]

    yy = np.zeros(yyl)
    yy[(y / ti).astype(int)] = 1
    yy = np.r_[yy,np.zeros(xxl-1)]

    XX = scifft.fft(xx)
    YY = scifft.fft(yy)
    XY = XX * np.conj(YY)
    xy = np.real(scifft.ifft(XY))/(N*ti)
    if standard:
        tx, cx = interval_correlate(x,x,standard=False)
        ty, cy = interval_correlate(y,y,standard=False)
        xy /= np.sqrt(cx[tx==0])*np.sqrt(cy[ty==0])
    return t, xy

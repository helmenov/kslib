## Fourier Integralを利用したFourier Series Expansion

import numpy as np
from scipy import integrate as sci_int
from matplotlib import pyplot as plt

def c_quad(func, a, b, **kwargs):
    """複素積分

    Args:
        func (_type_): Complex function
        a (_type_): lower bound
        b (_type_): upper bound
    """
    def real_func(t):
        return np.real(func(t))
    def imag_func(t):
        return np.imag(func(t))
    real_int = sci_int.quad(real_func, a, b, **kwargs)
    imag_int = sci_int.quad(imag_func, a, b, **kwargs)

    res = (real_int[0] + 1j * imag_int[0] , real_int[1:], imag_int[1:])
    return res

def fourier_integral(func, a, b, domain='time'):
    """Fourier Integral Function a.k.a. Continuous Fourier Transform Function

    Args:
        func (_type_): 被積分関数
        a (_type_): lower bound
        b (_type_): upper bound
        domain : 'time' / 'omega' (default:time)
    Output:
        Function : 変数omegaの関数
    """
    if domain == 'time': # CFT
        def _func(omega):
            def intee(t):
                return func(t) * np.exp(-1j * omega * t)
            return c_quad(intee, a, b)[0]
    elif domain == 'omega': # ICFT
        def _func(t):
            def intee(omega):
                return 1/(2*np.pi) * func(omega) * np.exp(1j * omega * t)
            return c_quad(intee, a, b)[0]
    return _func

# Fourier Series Expansion (FSE) for periodic function
class fourier_series_expansion_for_func():
    def __init__(self,func,period,complex_representation=True):
        """_summary_

        Args:
            func (_type_): Function
            period (_type_): 関数のみなし周期, 三角関数の基本周期
            complex_representation (bool, optional): 複素数係数形(c_n)ならTrue，余弦波用,正弦波用（a_n,b_n)ならFalse． Defaults to True.
        """
        self.func = func
        self.omega0 = 2*np.pi/period
        self.P = period
        self.c_mode = complex_representation

    def get_coeffs(self,N):
        """N次各周波数成分までの係数：いわゆるフーリエ変換

        Args:
            N (_type_): _description_

        Returns:
            _type_: _description_
        """
        coeffs = list()
        omegas = list()
        def _func(t):
            return self.func(t,self.P)
        CFT = fourier_integral(_func, 0, self.P, domain='time')
        if self.c_mode:
            for n in range(-N,N+1):
                omega_n = n * self.omega0
                c_n = (1/self.P) * CFT(omega_n)
                omegas.append(omega_n)
                coeffs.append(c_n)
        else:
            for n in range(N+1):
                omega_n = n * self.omega0
                def intee_an(t):
                    return self.func(t,self.P) * np.cos(omega_n * t)
                def intee_bn(t):
                    return self.func(t,self.P) * np.sin(omega_n * t)

                a_n = (2/self.P) * sci_int.quad(intee_an, 0, self.P)[0]
                b_n = (2/self.P) * sci_int.quad(intee_bn, 0, self.P)[0]
                omegas.append(omega_n)
                coeffs.append((a_n, b_n))

        return np.array(omegas), np.array(coeffs)

    def fit(self, omegas, coeffs):
        """omegas, coeffsによるフーリエ近似: これがいわゆるフーリエ級数展開
        (* 一般にはフーリエ「逆」変換)

        Args:
            omegas (_type_): _description_
            coeffs (_type_): _description_
        """
        def _func(t):
            if self.c_mode:
                res = 0 + 0j
                for omega_n, c_n in zip(omegas,coeffs):
                    res += c_n * np.exp(1j * omega_n * t)
            else:
                res = 0
                for n, omega_n in enumerate(omegas):
                    a_n,b_n = coeffs[n]
                    if n > 0:
                        res += a_n * np.cos(omega_n * t) + b_n * np.sin(omega_n * t)
                    elif n==0:
                        res += a_n/2
            return res
        return _func


# Fourier Series Expansion for sampled finite-length signal
# つまり，離散時間フーリエ変換 discrete-time fourie transform
class fourier_series_expansion_for_signal():
    def __init__(self,signal,Fs,complex_representation=True):
        """_summary_

        Args:
            signal (_type_): signal
            Fs (_type_): signalのサンプリング周波数
            complex_representation (bool, optional): 複素数係数形(c_n)ならTrue，余弦波用,正弦波用（a_n,b_n)ならFalse． Defaults to True.
        """
        self.signal = signal
        self.L = len(self.signal)
        self.Ts = 1/Fs
        self.P = self.L*self.Ts
        self.omega0 = 2*np.pi/self.P
        self.c_mode = complex_representation

    def get_coeffs(self,N):
        """N次各周波数成分までの係数

        Args:
            N (_type_): _description_

        Returns:
            _type_: _description_
        """
        coeffs = list()
        omegas = list()
        t = np.arange(self.L)*self.Ts
        if self.c_mode:
            for n in range(-N,N+1):
                omega_n = n * self.omega0
                c_n = (1/self.L) * (self.signal * np.exp(-1j * omega_n * t)).sum()
                omegas.append(omega_n)
                coeffs.append(c_n)
        else:
            for n in range(N+1):
                omega_n = n * self.omega0
                a_n = (2/self.L) * (self.signal * np.cos(omega_n * t)).sum()
                b_n = (2/self.L) * (self.signal * np.sin(omega_n * t)).sum()
                omegas.append(omega_n)
                coeffs.append((a_n, b_n))

        return np.array(omegas), np.array(coeffs)

    def fit(self, omegas, coeffs):
        """omegas, coeffsによるフーリエ近似

        Args:
            omegas (_type_): _description_
            coeffs (_type_): _description_
        """
        def _func(t):
            if self.c_mode:
                res = 0 + 0j
                for omega_n, c_n in zip(omegas,coeffs):
                    res += c_n * np.exp(1j * omega_n * t)
            else:
                res = 0
                for n, omega_n in enumerate(omegas):
                    a_n,b_n = coeffs[n]
                    if n > 0:
                        res += a_n * np.cos(omega_n * t) + b_n * np.sin(omega_n * t)
                    elif n==0:
                        res += a_n/2
            return res
        return _func


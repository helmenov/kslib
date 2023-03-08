import numpy as np
from scipy import fftpack as scifft

class fourierAnalysis():
    def __init__(self, x):
        X = scifft.fft(x)
        self.__X = X

        A = np.abs(X)
        self.__amp = A

        logA = np.log(np.array([a.clip(min=1e-7) for a in A]))
        self.__dB = 20 * logA

        Phi = np.unwrap(np.angle(X))
        self.__phase = Phi

        C_amp = scifft.ifft(logA)
        self.__C_amp = C_amp

        jPhi = np.array([complex(0,p) for p in Phi])
        C_phase = scifft.ifft(jPhi)
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


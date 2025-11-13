import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

T = 128
a = 0.05
n = np.arange(T)
s = np.exp(-a * (n + 1))

S_fft = fftshift(fft(s))
k = np.arange(-T//2, T//2)

plt.figure()
plt.plot(n, s)
plt.title("s[n] = exp(-a n)")

plt.figure()
plt.stem(k, np.abs(S_fft), basefmt=" ")
plt.xlabel("k")
plt.ylabel("|S(k)| (FFT)")
plt.title("FFT spectrum of s[n]")

S_dft = np.zeros(T, dtype=complex)
for ki in range(T):
    S_dft[ki] = np.sum(s * np.exp(-1j * 2 * np.pi * n * ki / T))
S_dft_shift = fftshift(S_dft)

plt.figure()
plt.stem(k, np.abs(S_dft_shift), basefmt=" ")
plt.xlabel("k")
plt.ylabel("|S(k)| (DFT)")
plt.title("DFT spectrum of s[n]")

fc = 20 / T
x = (1 + s) * np.cos(2 * np.pi * fc * n)

plt.figure()
plt.plot(n, x)
plt.title("x[n] = (1 + s[n]) cos(2π f_c n)")

X_fft = fftshift(fft(x))

plt.figure()
plt.stem(k, np.abs(X_fft), basefmt=" ")
plt.xlabel("k")
plt.ylabel("|X(k)| (FFT)")
plt.title("FFT spectrum of x[n]")

X_dft = np.zeros(T, dtype=complex)
for ki in range(T):
    X_dft[ki] = np.sum(x * np.exp(-1j * 2 * np.pi * n * ki / T))
X_dft_shift = fftshift(X_dft)

plt.figure()
plt.stem(k, np.abs(X_dft_shift), basefmt=" ")
plt.xlabel("k")
plt.ylabel("|X(k)| (DFT)")
plt.title("DFT spectrum of x[n]")

plt.show()

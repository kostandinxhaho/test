import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift

Fs = 8000.0
N = 64
t = np.arange(N) / Fs
fc = 3000.0
signal = np.cos(2 * np.pi * fc * t)

spectrum = fft(signal, n=N)

f_0Fs = np.arange(N) * (Fs / N)

plt.figure()
plt.stem(f_0Fs, np.abs(spectrum), basefmt=" ")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Spectrum of signal")
plt.grid(True)

spec_shift = fftshift(spectrum)
f_centered = np.linspace(-Fs/2, Fs/2, N, endpoint=False)

plt.figure()
plt.stem(f_centered, np.abs(spec_shift), basefmt=" ")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Zero-centered frequency spectrum of signal")
plt.grid(True)

plt.show()

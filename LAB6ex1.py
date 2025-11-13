import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

npz = np.load('C:/Users/Kosta/Desktop/PROJEKT/PS/noisy_signal.npz')
x = npz['noisy_signal'].astype(float)
N = x.size
fs = N
t = np.arange(N) / fs

S = np.fft.fft(x)
S_shift = np.fft.fftshift(S)
k = np.arange(-N//2, N//2)


k_pos = np.arange(0, N//2 + 1)

S_mag_pos = np.abs(S[:N//2+1])

keep_mask = (np.abs(k) <= 9) 
S_keep_shift = S_shift * keep_mask
S_keep = np.fft.ifftshift(S_keep_shift)


x_rec_ifft = np.fft.ifft(S_keep).real


kept_bins = np.where(np.abs(k) <= 9)[0]
k_kept = k[kept_bins]     
S_kept_vals = S_shift[kept_bins]


n = np.arange(N)
E = np.exp(1j * 2 * np.pi * np.outer(k_kept, n) / N)
x_rec_idft = (1.0 / N) * (S_kept_vals[:, None] * E).sum(axis=0)
x_rec_idft = x_rec_idft.real

total_power = (1.0 / N) * np.sum(np.abs(S)**2)
signal_power = (1.0 / N) * np.sum(np.abs(S_keep)**2)
noise_power = total_power - signal_power
SNR = signal_power / noise_power
SNR_dB = 10.0 * np.log10(SNR)

print(f"N = {N}, fs = {fs} Hz")
print(f"Signal power (|k|<=9): {signal_power:.6f}")
print(f"Noise power:            {noise_power:.6f}")
print(f"SNR (linear):           {SNR:.6f}")
print(f"SNR (dB):               {SNR_dB:.2f} dB")

plt.figure()
plt.plot(t, x, linewidth=1)
plt.title('Noisy signal (time domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

plt.figure()
freqs_pos = (fs / N) * k_pos
plt.stem(freqs_pos, S_mag_pos, basefmt=" ")
plt.title('Frequency spectrum (positive side)')
plt.xlabel('DFT frequency [Hz]')
plt.ylabel('|S[k]|')
plt.grid(True, alpha=0.3)

plt.figure()
plt.plot(t, x, label='Noisy', linewidth=1)
plt.plot(t, x_rec_ifft, label='Reconstructed (ifft)', linewidth=1.5)
plt.plot(t, x_rec_idft, '--', label='Reconstructed (IDFT formula)', linewidth=1)
plt.title('Reconstructed signal after removing high freqs')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

npz = np.load('C:/Users/Kosta/Desktop/PROJEKT/PS/lab6_noisy_sound.npz')

s  = npz['noisy_sound'].astype(float)
fs = int(npz['fs'])
N  = s.size
t  = np.arange(N) / fs

S = np.fft.fft(s)
freqs = np.fft.fftfreq(N, d=1.0/fs)

pos_mask  = freqs >= 0
freqs_pos = freqs[pos_mask]
S_mag_pos = np.abs(S[pos_mask])

band_mask = (np.abs(freqs) <= 500.0)
S_keep = S * band_mask

s_clean = np.fft.ifft(S_keep).real

total_power  = (1.0 / N) * np.sum(np.abs(S)**2)
signal_power = (1.0 / N) * np.sum(np.abs(S_keep)**2)
noise_power  = total_power - signal_power
SNR_linear   = signal_power / max(noise_power, 1e-12)
SNR_dB       = 10.0 * np.log10(SNR_linear)

print(f"N = {N}, fs = {fs} Hz")
print("Signal band: 0–500 Hz")
print(f"Signal power: {signal_power:.6f}")
print(f"Noise power:  {noise_power:.6f}")
print(f"SNR (lin):    {SNR_linear:.6f}")
print(f"SNR (dB):     {SNR_dB:.2f} dB")

plt.figure()
plt.plot(t, s, linewidth=0.8)
plt.title('Noisy sound (time domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)

plt.figure()
markerline, stemlines, baseline = plt.stem(freqs_pos, S_mag_pos, basefmt=" ")
plt.xlim(0, fs/2)
plt.title('Positive frequency spectrum of noisy sound')
plt.xlabel('Frequency [Hz]')
plt.ylabel('|S[k]|')
plt.grid(True, alpha=0.3)

plt.figure()
plt.plot(t, s, label='Noisy', linewidth=0.8)
plt.plot(t, s_clean, label='Clean (0–500 Hz kept)', linewidth=1.2)
plt.title('Reconstructed signal after low-pass (0–500 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

try:
    from scipy.io.wavfile import write
    s_out = (s_clean / max(np.max(np.abs(s_clean)), 1e-9) * 32767).astype(np.int16)
    write('speech_clean.wav', fs, s_out)
    print("Saved cleaned audio to 'speech_clean.wav'")
except Exception as e:
    print(f"(Skipping save: {e})")

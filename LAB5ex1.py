import numpy as np
import matplotlib.pyplot as plt

Fs = 128.0
N = 128
t_max = 1.0
t = np.linspace(0, t_max, int(N), endpoint=False)

freqs = [1, 2, 10, 20, 100]
A = 1.0

M = 5
h = np.ones(M) / M 

def ma_filter(x, h):
    """Filtru medie mobilă: y[n] = (1/M) * sum_{k=0..M-1} x[n-k]"""
    return np.convolve(x, h, mode='full')[:len(x)]

signals = {}
outputs = {}

for f in freqs:
    x = A * np.sin(2 * np.pi * f * t)
    y = ma_filter(x, h)
    signals[f] = x
    outputs[f] = y

for f in freqs:
    plt.figure()
    plt.plot(t, signals[f], label=f"x[n], f={f} Hz", color='orange')
    plt.plot(t, outputs[f], label="y[n] (5-pt MA)", color='blue')
    plt.title(f"Input vs Output — {f} Hz sinusoid (Fs={Fs:.0f} Hz, N={N})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
plt.show()

np.random.seed(0)
f_example = 10
x_clean = signals[f_example]
noise = 0.1 * np.random.randn(N)
x_noisy = x_clean + noise
y_noisy = ma_filter(x_noisy, h)

plt.figure()
plt.plot(t, x_noisy, label="Input (noisy)", color='orange')
plt.plot(t, y_noisy, label="Output after 5-pt MA", color='blue')
plt.title("Noisy example (σ=0.1) — 10 Hz sinusoid")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

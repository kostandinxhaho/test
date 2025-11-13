import numpy as np
import matplotlib
import matplotlib.pyplot as plt

T, A, N = 100, 3, 1000
t = np.linspace(0, T, N, endpoint=False)

s = np.ones_like(t) * A
s[t >= T/2] = -A

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t, s, 'b')
plt.grid(True, alpha=0.3)
plt.title("Original rectangular signal")
plt.xlabel("Time"); plt.ylabel("Amplitude")
plt.ylim([-A - 1, A + 1])

kmax = 9
ks = np.arange(-kmax, kmax + 1)
ck = np.zeros(len(ks), dtype=complex)
for i, k in enumerate(ks):
    if k == 0: ck[i] = 0
    elif k % 2 != 0: ck[i] = (2*A)/(1j*np.pi*k)
    else: ck[i] = 0

plt.subplot(3, 1, 2)
plt.stem(ks, np.abs(ck), basefmt=" ", markerfmt='o', linefmt=':')
plt.title("Fourier coefficients magnitude")
plt.xlabel("Frequency index k"); plt.ylabel("|c_k|")

s_recon = np.zeros_like(t, dtype=complex)
for i, k in enumerate(ks):
    s_recon += ck[i] * np.exp(1j * 2*np.pi * k * t / T)
s_recon = np.real(s_recon)

plt.subplot(3, 1, 3)
plt.plot(t, s, 'k', label='original')
plt.plot(t, s_recon, 'r--', label='reconstructed')
plt.grid(True, alpha=0.3)
plt.title(f"Reconstructed signal with kmax = {kmax}")
plt.xlabel("Time"); plt.ylabel("Amplitude")
plt.ylim([-A - 1, A + 1]); plt.legend()

plt.tight_layout()
plt.savefig("fourier_rect.png", dpi=150)
plt.show(block=True)

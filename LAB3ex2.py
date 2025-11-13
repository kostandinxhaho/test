import numpy as np
import matplotlib.pyplot as plt

T = 100
A = 1
N_samples = 1000
t = np.linspace(0, T, N_samples, endpoint=False)

s = np.ones_like(t) * A
s[t >= T/2] = -A

plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(t, s, 'b')
plt.grid(True, alpha=0.3)
plt.title("Original rectangular signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.ylim([-A - 0.5, A + 0.5])

Kmax = 500
ks = np.arange(-Kmax, Kmax + 1)
ck = np.zeros(len(ks), dtype=complex)

for i, k in enumerate(ks):
    if k == 0:
        ck[i] = 0
    elif k % 2 != 0:
        ck[i] = (2 * A) / (1j * np.pi * k)
    else:
        ck[i] = 0

plt.subplot(3, 1, 2)
plt.stem(ks, np.abs(ck), basefmt=" ", markerfmt='o', linefmt=':')
plt.grid(True, alpha=0.3)
plt.title("Fourier coefficients magnitude |c_k|")
plt.xlabel("Frequency index k")
plt.ylabel("|c_k|")
plt.xlim([-50, 50])

total_energy = np.sum(np.abs(ck)**2)

Ns = np.arange(1, 101)
rms_errors = []

for N in Ns:
    idx_keep = (ks >= -N) & (ks <= N)
    energy_used = np.sum(np.abs(ck[idx_keep])**2)
    rms_err = np.sqrt(total_energy - energy_used)
    rms_errors.append(rms_err)


plt.subplot(3, 1, 3)
plt.plot(Ns, rms_errors, 'r')
plt.grid(True, alpha=0.3)
plt.title("RMS error as function of N")
plt.xlabel("N (number of coefficients used on each side)")
plt.ylabel("RMS error")
plt.tight_layout()
plt.show()

threshold = 0.05
N_opt = Ns[np.where(np.array(rms_errors) < threshold)[0][0]]
print(f"Smallest N with RMS error < {threshold}: N_opt = {N_opt}")

ck_recon = np.zeros_like(ck, dtype=complex)
idx_keep_opt = (ks >= -N_opt) & (ks <= N_opt)
ck_recon[idx_keep_opt] = ck[idx_keep_opt]

s_recon = np.zeros_like(t, dtype=complex)
for i, k in enumerate(ks):
    s_recon += ck_recon[i] * np.exp(1j * 2 * np.pi * k * t / T)
s_recon = np.real(s_recon)

plt.figure(figsize=(10, 4))
plt.plot(t, s, 'k', label='original')
plt.plot(t, s_recon, 'r--', label=f'reconstructed (N={N_opt})')
plt.grid(True, alpha=0.3)
plt.title(f"Reconstruction with N = {N_opt} coefficients")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.ylim([-A - 0.5, A + 0.5])
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

A = 1.0
T = 100.0
# time delay
tau = T / 4
# highest harmonic index                
K = 81
# arranging k from -81 to 81              
ks = np.arange(-K, K + 1)    
w0 = 2 * np.pi / T

# Original rectangular signal over one period [0, T)
t = np.linspace(0, T, 4000, endpoint=False)
s = np.where(t < T / 2, A, -A)

# Fourier coefficients c_k
ck = np.zeros_like(ks, dtype=complex)
for i, k in enumerate(ks):
    if k == 0:
        # DC term is zero
        ck[i] = 0.0                      
    elif k % 2 != 0:
        ck[i] = 2 * A / (1j * np.pi * k)
    else:
        ck[i] = 0.0

# Plot |ck|
plt.figure()
plt.stem(ks, np.abs(ck), basefmt=" ")
plt.title(r'|$c_k$| (original)')
plt.xlabel('k')
plt.ylabel('|c_k|')
plt.grid(True)

# Apply delay:
ck_del = ck * np.exp(-1j * 2 * np.pi * ks * tau / T)

# Phase before and after
phase_before = np.angle(ck, deg=True)
phase_after = np.angle(ck_del, deg=True)

# Plot both phases on the same figure using stem
plt.figure()

# Stem for before delay
(markerline1, stemlines1, baseline1) = plt.stem(
    ks, phase_before, basefmt=" ", linefmt='C0-', markerfmt='C0o', label='Before delay'
)
plt.setp(stemlines1, 'linewidth', 1)

# Stem for after delay
(markerline2, stemlines2, baseline2) = plt.stem(
    ks, phase_after, basefmt=" ", linefmt='C1-', markerfmt='C1o', label='After delay'
)
plt.setp(stemlines2, 'linewidth', 1)

plt.title('Phase of $c_k$ (before and after delay)')
plt.xlabel('k')
plt.ylabel('degrees')
plt.legend()
plt.grid(True)




#Reconstruct from modified spectrum
exp_matrix = np.exp(1j * np.outer(ks, w0 * t))
s_rec = np.real(np.dot(ck_del, exp_matrix))

# true shifted version of s(t)
def shift_periodic(shift, T, tgrid):
    return np.where(((tgrid - shift) % T) < T / 2, A, -A)

s_shift_true = shift_periodic(tau, T, t)

# Plot original
plt.figure()
plt.plot(t, s, label='original s(t)', linewidth=1)
plt.plot(t, s_shift_true, '--', label='true s(t - τ)', linewidth=1)
plt.plot(t, s_rec, label='reconstructed from $c_k$ after delay', linewidth=1)
plt.xlim(0, T)
plt.ylim(-1.5, 1.5)
plt.xlabel('t')
plt.ylabel('amplitude')
plt.title('Reconstruction check')
plt.legend()
plt.grid(True)

plt.show()

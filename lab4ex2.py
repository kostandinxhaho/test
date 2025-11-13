import numpy as np
import matplotlib.pyplot as plt

# number of samples per period
T = 100                
A = 1.0
# pulse width
Delta = T / 5 
# number of Fourier coefficients         
N = 31                 
ks = np.arange(0, N + 1)
t = np.linspace(0, T, 4000, endpoint=False)

# Pulse of width Delta starting at t=0
s = np.where(t < Delta, A, 0)

plt.figure()
plt.plot(t, s)
plt.title('Original pulse signal')
plt.xlabel('t')
plt.ylabel('amplitude')
plt.grid(True)

# Compute Fourier coefficients c_k 
ck = np.zeros_like(ks, dtype=complex)
for i, k in enumerate(ks):
    if k == 0:
        ck[i] = A * (Delta / T)
    else:
        ck[i] = A * np.exp(-1j * np.pi * k * Delta / T) * (Delta / T) * np.sinc(k * Delta / T)

plt.figure()
plt.stem(ks, np.abs(ck), basefmt=" ")
plt.title(r'|$c_k$| (input signal)')
plt.xlabel('k')
plt.ylabel('|c_k|')
plt.grid(True)

# Define transfer function
def H(f, fc):
    """Low-pass RC filter transfer function"""
    return 1 / (1 + 1j * 2 * np.pi * f / fc)

# Frequencies associated with coefficients
f_k = ks / T

fc_values = [0.1, 1, 10]

# Filter and reconstruct 
for fc in fc_values:
    Hk = H(f_k, fc)
    ck_y = ck * Hk

    # Reconstruct signal using Fourier series
    exp_matrix = np.exp(1j * 2 * np.pi * np.outer(ks, t) / T)
    y_t = np.real(np.dot(ck_y, exp_matrix) +
                  np.dot(np.conj(ck_y[1:]), np.exp(-1j * 2 * np.pi * np.outer(ks[1:], t) / T)))

    # Plot |c_k^y|
    plt.figure()
    plt.stem(ks, np.abs(ck_y), basefmt=" ")
    plt.title(rf'|$c_k^y$| after LPF with $f_c$ = {fc}')
    plt.xlabel('k')
    plt.ylabel('|c_k^y|')
    plt.grid(True)

    # Plot reconstructed signal
    plt.figure()
    plt.plot(t, s, label='original pulse', linewidth=1)
    plt.plot(t, y_t, label=f'filtered (fc={fc})', linewidth=1)
    plt.xlim(0, T)
    plt.title(f'Filtered signal for fc = {fc}')
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.legend()
    plt.grid(True)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

def ramp(N: int) -> np.ndarray:
    """Discrete ramp of length N: r(i) = i for i = 0..N-1"""
    y = np.zeros(N, dtype=int)
    for t in range(N):
        y[t] = t
    return y

def ustep(N: int) -> np.ndarray:
    """Discrete unit step of length N: u(i) = 1 for i = 0..N-1"""
    return np.ones(N, dtype=int)

def delay(x: np.ndarray, T: int) -> np.ndarray:
    """Delay sequence x by T samples with zero padding, keep length len(x)."""
    if T <= 0:
        return x.copy()
    return np.pad(x[:-T], (T, 0), mode="constant", constant_values=0)

N = 200
T = 100

r  = ramp(N)
u  = ustep(N)
rT = delay(r, T)
uT = delay(u, T)

s = r - rT - T * uT

i = np.arange(N)

plt.figure(figsize=(8,5))
plt.plot(i, s,                 label="sum", linewidth=2)
plt.plot(i, r,                 label="r(t)")
plt.plot(i, -rT,               label="-r(t-T)")
plt.plot(i, -T * uT,           label=r"-T·u(t-T)")

plt.title("Signals")
plt.xlabel("Points")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

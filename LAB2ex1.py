import numpy as np
import matplotlib.pyplot as plt
import math

f1, f2 = 1600, 1800
g = math.gcd(f1, f2)
T = 1 / g 


fs = 200_000
Ns = int(T * fs)

t1 = np.arange(0, Ns) / fs
t2 = np.arange(Ns, 2*Ns) / fs
t3 = np.arange(2*Ns, 3*Ns) / fs

def x(t, A1, A2):
    return A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t)

y = np.concatenate([
    x(t1, 0, 1),
    x(t2, 1, 1),
    x(t3, 1, 0)
])
t = np.concatenate([t1, t2, t3])

plt.figure(figsize=(8,3))
plt.plot(t, y, linewidth=1)
plt.title("Semnal produs de modem")
plt.xlabel("Time (s)")
plt.ylabel("Amplitudine")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

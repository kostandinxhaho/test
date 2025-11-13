import numpy as np
import matplotlib.pyplot as plt

N = 20
x = np.full(N, 60)
y = np.zeros(N, dtype=int)
f = np.zeros(N, dtype=int)
e = np.zeros(N, dtype=int)

y[0] = 7
f[0] = 0
e[0] = 0 

def S1(y_cur: int, err: int) -> int:
    if err > 10:
        return y_cur + 5
    elif err > 0:
        return y_cur + 1
    else:
        return y_cur


for i in range(N - 1):
    e[i] = x[i] - f[i]
    y[i + 1] = S1(y[i], e[i])
    f[i + 1] = y[i + 1]

e[-1] = x[-1] - f[-1]

print("y(t) =", list(y))

t = np.arange(N)
plt.figure(figsize=(7,4))
plt.step(t, x, where='post', label='x(t) (desired)', linewidth=2)
plt.step(t, y, where='post', label='y(t) (measured)', linewidth=2)
plt.title("Discrete Feedback Cruise Control (Toy Model)")
plt.xlabel("Step")
plt.ylabel("Speed")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

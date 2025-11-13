import numpy as np
import matplotlib.pyplot as plt


t = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

s1 = np.exp(1j * t)
x1 = np.real(s1)
y1 = np.imag(s1)


s2 = np.exp(-1j * t)
x2 = np.real(s2)
y2 = np.imag(s2)


s_cos = (s1 + s2) / 2
x_cos = np.real(s_cos)
y_cos = np.imag(s_cos)


print("Computed cos(t):", x_cos)
print("numpy cos(t):   ", np.cos(t))


s_sin = (s1 - s2) / (2j)
print("Computed sin(t):", np.real(s_sin))
print("numpy sin(t):   ", np.sin(t))

plt.figure(figsize=(6,6))
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

plt.plot(x1, y1, 'ro', label='e^{j t}')
plt.plot(x2, y2, 'bo', label='e^{-j t}')
plt.plot(x_cos, y_cos, 'go', label='cos(t)')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Euler’s Formula - Complex Plane')
plt.grid(True)
plt.axis('equal')
plt.show()

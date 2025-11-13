import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import time

# 1)
print("=== 1) Basic arithmetic ===")
a = 2
b = 3
print("a + b =", a + b)
c = a + b
print("c =", c)
print("a * b =", a * b)
print("a / b =", a / b)
print("b / a =", b / a)
print("a ** b =", a ** b)

# 2)
print("\n=== 2) Vectors ===")
vec1 = np.array([1, 2, 3, 4])
vec2 = np.array([1, 2, 3, 4])
print("vec1:", vec1)
print("vec2:", vec2)

vec3 = np.arange(1, 8)
k = 10
vec3_scaled = k * vec3
print("vec3:", vec3)
print(f"{k} * vec3:", vec3_scaled)

rng = np.random.default_rng(seed=42)
vec_rand = rng.random(5)
print("vec_rand:", vec_rand)
print("vec1[0] =", vec1[0], " vec1[2] =", vec1[2])
print("vec1[1:3] =", vec1[1:3], " vec1[-2:] =", vec1[-2:])
print("vec1 dtype:", vec1.dtype, " shape:", vec1.shape)
print("vec_rand dtype:", vec_rand.dtype, " shape:", vec_rand.shape)

# 3)
print("\n=== 3) Matrices ===")
M1 = rng.random((5, 4))
M2 = np.ones((5, 4))
M_sum = M1 + M2
print("M1:\n", M1)
print("\nM2:\n", M2)
print("\nM1 + M2:\n", M_sum)

# 4)
print("\n=== 4) Transpose ===")
M1_T = M1.T
print("M1.T shape:", M1_T.shape)

col_vec = np.ones((7, 1))
print("\nColumn vector (7x1):\n", col_vec, "\nshape:", col_vec.shape)
row_vec = col_vec.T
print("\nTranspose of column vector (1x7):\n", row_vec, "\nshape:", row_vec.shape)
print("Difference: column (7,1) vs row (1,7)")

# 5)
print("\n=== 5) Sequences ===")
seq_a = np.arange(5, 5 + 10)
seq_b = np.arange(5, 5 + 10 * 3, 3)
seq_c = np.linspace(5, 14, 10)
print("seq_a:", seq_a)
print("seq_b:", seq_b)
print("seq_c:", seq_c)
print("Length seq_a:", len(seq_a), " shape:", seq_a.shape)
print("Dimensions M1:", M1.shape, " M1.T:", M1_T.shape)

scaled_columns = M1 * np.arange(1, M1.shape[1] + 1)
print("\nBroadcasting (M1 * [1,2,3,4]):\n", scaled_columns)

# 6)
print("\n=== 6) Strings ===")
s = "Signal processing"
print(s)
combined = "lab 0 - " + s
print(combined)
n = 7
print(f"Exercise {n}: {s}")

# 7)
print("\n=== 7) Plots ===")
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)
f1 = 1
f2 = 2
y1 = np.sin(2 * np.pi * f1 * t)
y2 = np.sin(2 * np.pi * f2 * t)

plt.figure(figsize=(8, 4))
plt.plot(t, y1, label="sin 1 Hz")
plt.plot(t, y2, linestyle='--', label="sin 2 Hz")
plt.title("Sinusoids 1 Hz and 2 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("sinusoids.png", dpi=150)
plt.show()

# 8)
print("\n=== 8) Simple signal processing ===")
y_sum = y1 + y2

plt.figure()
plt.plot(t, y1)
plt.title("Sinusoid 1 Hz")
plt.grid(True)

plt.figure()
plt.plot(t, y2)
plt.title("Sinusoid 2 Hz")
plt.grid(True)

plt.figure()
plt.plot(t, y_sum)
plt.title("Sum of 1 Hz and 2 Hz sinusoids")
plt.grid(True)
plt.show()

# 9)
print("\n=== 9) Image denoising ===")
img_names = ["Img_initial.png", "R1.png", "R2.png"]
loaded_imgs = []

for name in img_names:
    if os.path.exists(name):
        img = imread(name).astype(float)
        loaded_imgs.append(img)
        plt.figure()
        plt.imshow(img.astype(np.uint8))
        plt.title(f"Image: {name}")
        plt.axis('off')
    else:
        print(f"⚠️ File {name} not found — skipping.")

if len(loaded_imgs) == 3:
    I0, R1, R2 = loaded_imgs
    IR = I0 * 0.3 + R1 * 0.3 + R2 * 0.3
    plt.figure()
    plt.imshow(IR.astype(np.uint8))
    plt.title("Reconstructed image")
    plt.axis('off')
    plt.show()
else:
    print("Not all three images found — reconstruction skipped.")

# 10)
print("\n=== 10) Matrix multiplication ===")
A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
print("Matrix A:\n", A)
print("Matrix B:\n", B)

C_hadamard = A * B
print("\nHadamard product (A * B):\n", C_hadamard)

C_dot = np.dot(A, B)
print("\nDot product (np.dot):\n", C_dot)

C_matmul = A @ B
print("\nMatrix multiplication (A @ B):\n", C_matmul)

# 11)
print("\n=== 11) Audio signal processing ===")
audio_file = "noisy_signal.npz"
if os.path.exists(audio_file):
    npz = np.load(audio_file)
    noisy_signal = npz['noisy_sound']
    fs_signal = noisy_signal.shape[0]
    print("Signal length:", fs_signal, "samples")

    t_sig = np.arange(fs_signal)
    plt.figure(figsize=(10, 4))
    plt.plot(t_sig, noisy_signal)
    plt.title("Noisy audio signal")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️ File {audio_file} not found — skipping audio processing.")

# 12)
print("\n=== 12) Row-major vs column-major vs vectorized ===")
N = 1000
A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))

t1 = time.time()
for i in range(N):
    for j in range(N):
        C[i, j] = A[i, j] * B[i, j]
t_row = time.time() - t1

t2 = time.time()
for j in range(N):
    for i in range(N):
        C[i, j] = A[i, j] * B[i, j]
t_column = time.time() - t2

t3 = time.time()
C = A * B
t_vectorized = time.time() - t3

print(f"Timp row-major:       {t_row} s")
print(f"Timp column-major:    {t_column} s")
print(f"Timp matrix operation: {t_vectorized} s")

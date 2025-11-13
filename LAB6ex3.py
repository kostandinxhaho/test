import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

IMG_PATH = 'PS/peppers.png'
rgb = image.imread(IMG_PATH)

plt.figure()
plt.imshow(rgb, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title("Original Image")
plt.axis('off')
plt.show(block=False)

def rgb2gray(rgb_arr):
    arr = rgb_arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])

img = rgb2gray(rgb)
img = np.clip(img, 0.0, 1.0)

S = fft2(img)
S_c = fftshift(S)

def spec_vis(S_):
    return np.log1p(np.abs(S_))

plt.figure()
plt.imshow(spec_vis(S_c), cmap='gray')
plt.title('Centered spectrum (log magnitude)')
plt.axis('off')
plt.show(block=False)


H, W = img.shape
yy, xx = np.ogrid[:H, :W]
cy, cx = H//2, W//2
cut_frac = 0.12
R = int(min(H, W) * cut_frac)

dist2 = (yy - cy)**2 + (xx - cx)**2
low_mask = dist2 <= (R**2)

S1_c = S_c * low_mask
S2_c = S_c * (~low_mask)

S1 = ifftshift(S1_c)
S2 = ifftshift(S2_c)

reconstructed_low  = np.real(ifft2(S1))
reconstructed_high = np.real(ifft2(S2))

def to01(x):
    x = x - x.min()
    d = x.max()
    return x / d if d > 0 else x

reconstructed_low_disp  = to01(reconstructed_low)
reconstructed_high_disp = to01(reconstructed_high)

plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
plt.title('Grayscale, normalized')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(spec_vis(S_c), cmap='gray')
plt.title('Centered Spectrum (log)')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(low_mask, cmap='gray')
plt.title(f'Low-pass mask (R={R}px)')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(reconstructed_low_disp, cmap='gray', vmin=0, vmax=1)
plt.title('Reconstructed (low-freqs)')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(spec_vis(S1_c), cmap='gray')
plt.title('Low spectrum (log)')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(reconstructed_high_disp, cmap='gray', vmin=0, vmax=1)
plt.title('Reconstructed (high-freqs)')
plt.axis('off')

plt.tight_layout()
plt.show()

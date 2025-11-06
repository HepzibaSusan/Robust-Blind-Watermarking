import cv2
import numpy as np
from scipy.fftpack import dct

# ---- 1. Load the watermarked image ----
wm_path = r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_rot5.png"
img = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
img = np.float32(img)

# ---- 2. Same parameters as embedding ----
block_size = 8
mid_band_coords = [(2,1),(1,2),(2,2),(3,1),(1,3)]

# ---- 3. Extract bits ----
h, w = img.shape
extracted_bits = []

for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = img[i:i+block_size, j:j+block_size]
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # Only read as many bits as we originally embedded (5)
        if len(extracted_bits) < 5:
            # Average of our chosen mid-band coefficients
            vals = [dct_block[r, c] for (r, c) in mid_band_coords]
            mean_val = np.mean(vals)

            # Decide bit: positive shift → 1, negative shift → 0
            bit = 1 if mean_val > 0 else 0
            extracted_bits.append(bit)

print("Extracted watermark bits:", extracted_bits)

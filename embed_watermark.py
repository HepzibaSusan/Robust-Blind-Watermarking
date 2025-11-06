import cv2
import numpy as np
from scipy.fftpack import dct, idct

# --- 1. Load the image ---
img = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\data\cameraman.png",
                 cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))        
img = np.float32(img)

# --- 2. Prepare a tiny watermark (bits 0/1) ---
watermark_bits = np.array([1,0,0,1,1], dtype=np.uint8)
mid_band_coords = [(2,1),(1,2),(2,2),(3,1),(1,3)]  # from the previous run

# --- 3. Split image into 8x8 blocks ---
h, w = img.shape
block_size = 8
watermarked = np.zeros_like(img)

bit_idx = 0
for i in range(0, h, block_size):
    for j in range(0, w, block_size):
        block = img[i:i+block_size, j:j+block_size]

        # --- 4. Forward DCT ---
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # --- 5. Embed one bit in the mid-band positions ---
        if bit_idx < len(watermark_bits):
            for (r, c) in mid_band_coords:
                # Slightly shift the coefficient up or down depending on bit
                if watermark_bits[bit_idx] == 1:
                    dct_block[r, c] += 5.0
                else:
                    dct_block[r, c] -= 5.0
            bit_idx += 1

        # --- 6. Inverse DCT ---
        block_idct = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
        watermarked[i:i+block_size, j:j+block_size] = block_idct

# --- 7. Clip and save ---
watermarked = np.clip(watermarked, 0, 255)
cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_image.png", np.uint8(watermarked))
print("Watermarked image saved to outputs/watermarked_image.png")

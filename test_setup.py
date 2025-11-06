# test_setup.py
import os
from utils import load_gray, dwt2_gray, idwt2_from_bands, blockwise_dct, blockwise_idct, detect_harris_points, overlay_points_and_save, mid_band_coords_8
import cv2
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.abspath(os.path.join(BASE, 'data'))
OUT = os.path.abspath(os.path.join(BASE, 'outputs'))
os.makedirs(OUT, exist_ok=True)

HOST = os.path.join(DATA, 'cameraman.png')  # put your host image here

print("Loading:", HOST)
img = load_gray(HOST)
print("Image shape:", img.shape)

# 1) DWT
LL, LH, HL, HH = dwt2_gray(img)
print("DWT shapes LL,LH,HL,HH:", LL.shape, LH.shape, HL.shape, HH.shape)

# save LH and HL for inspection
from utils import save_gray
save_gray(os.path.join(OUT, 'LL.png'), LL)
save_gray(os.path.join(OUT, 'LH.png'), LH)
save_gray(os.path.join(OUT, 'HL.png'), HL)
save_gray(os.path.join(OUT, 'HH.png'), HH)
print("Saved DWT bands to outputs/")

# 2) Blockwise DCT on LH
dct_LH = blockwise_dct(LH, block_size=8)
save_gray(os.path.join(OUT, 'dct_LH_vis.png'), np.log(np.abs(dct_LH)+1))
print("Saved DCT magnitude visualization.")

# 3) Feature detection (Harris)
# For feature detection we use the full-resolution grayscale image (uint8)
img_uint8 = np.clip(img,0,255).astype('uint8')
points = detect_harris_points(img_uint8, max_corners=200)
print("Detected Harris points:", len(points))
# save color overlay
img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
overlay_points_and_save(img_color, points, os.path.join(OUT, 'features_harris.png'))
print("Saved features overlay to outputs/features_harris.png")

# 4) Print mid-band coords sample
print("Mid-band coords (8x8):", mid_band_coords_8())

print("Test setup complete. Check outputs/ for images.")

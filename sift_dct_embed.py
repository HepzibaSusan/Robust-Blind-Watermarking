import cv2
import numpy as np
import os
import pickle

# --- Settings ---
img_path = r"C:\Users\91991\Desktop\robust_watermark\data\cameraman.png"
out_dir = r"C:\Users\91991\Desktop\robust_watermark\outputs"
os.makedirs(out_dir, exist_ok=True)

patch_size = 36
alpha = 50          # embedding strength
watermark_bits = [1, 0, 1, 1, 0]
redundancy = 10     # number of patches per bit
mid_band_coords = [(3,4),(2,3),(4,2),(3,3),(2,4)]

# --- Load host image ---
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
watermarked = img.astype(np.float32)
h, w = img.shape
half = patch_size // 2

# --- Detect SIFT keypoints ---
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# --- Filter keypoints near borders ---
keypoints_filtered = [k for k in keypoints if half <= k.pt[0] <= w-half and half <= k.pt[1] <= h-half]
keypoints = sorted(keypoints_filtered, key=lambda k: -k.response)

embed_info = []
kp_idx = 0

# --- Embed watermark bits ---
for bit in watermark_bits:
    for r in range(redundancy):
        if kp_idx >= len(keypoints):
            break
        kp = keypoints[kp_idx]
        x, y = int(kp.pt[0]), int(kp.pt[1])

        y0, y1 = y-half, y+half
        x0, x1 = x-half, x+half
        patch = watermarked[y0:y1, x0:x1]

        # Forward DCT
        dct_patch = cv2.dct(patch)

        # Embed bit in mid-band coefficients
        for coord in mid_band_coords:
            if bit == 1:
                dct_patch[coord] += alpha
            else:
                dct_patch[coord] -= alpha

        # Inverse DCT
        watermarked[y0:y1, x0:x1] = cv2.idct(dct_patch)
        embed_info.append({"pt":(x,y),"bit":bit})
        kp_idx += 1

# --- Save watermarked image ---
wm_uint8 = np.clip(watermarked,0,255).astype(np.uint8)
wm_path = os.path.join(out_dir,"watermarked_simple.png")
cv2.imwrite(wm_path, wm_uint8)
print("Watermarked image saved:", wm_path)

# --- Save embedding info for extraction ---
with open(os.path.join(out_dir,"embed_info_simple.pkl"),"wb") as f:
    pickle.dump(embed_info,f)
print("Embedding info saved.")

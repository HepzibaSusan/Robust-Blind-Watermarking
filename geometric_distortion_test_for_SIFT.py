import cv2
import numpy as np
import pickle
import os

# --- Paths ---
wm_image_path = r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_simple.png"
embed_info_path = r"C:\Users\91991\Desktop\robust_watermark\outputs\embed_info_simple.pkl"
out_dir = r"C:\Users\91991\Desktop\robust_watermark\outputs"
os.makedirs(out_dir, exist_ok=True)

# --- Load watermarked image ---
wm_img = cv2.imread(wm_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# --- Load embedding info ---
with open(embed_info_path,"rb") as f:
    embed_info = pickle.load(f)

# --- Extraction function ---
def extract_bits(img, embed_info, patch_size=36, mid_band_coords=[(3,4),(2,3),(4,2),(3,3),(2,4)], redundancy=10):
    half = patch_size // 2
    extracted_bits_all = []

    for info in embed_info:
        x, y = info["pt"]
        y0, y1 = y-half, y+half
        x0, x1 = x-half, x+half
        patch = img[y0:y1, x0:x1]
        if patch.shape[0]<8 or patch.shape[1]<8:
            continue
        dct_patch = cv2.dct(patch)
        energy = np.mean([dct_patch[c] for c in mid_band_coords])
        extracted_bits_all.append(1 if energy > 0 else 0)

    # Majority voting
    num_bits = len(extracted_bits_all)//redundancy
    final_bits = []
    for i in range(num_bits):
        votes = extracted_bits_all[i*redundancy:(i+1)*redundancy]
        final_bits.append(1 if votes.count(1)>votes.count(0) else 0)

    return final_bits

# --- Original ---
orig_bits = extract_bits(wm_img, embed_info)
print("Recovered from original:", orig_bits)

# --- Rotation (±10°) ---
angle = 10
M = cv2.getRotationMatrix2D((wm_img.shape[1]//2, wm_img.shape[0]//2), angle, 1.0)
rotated = cv2.warpAffine(wm_img, M, (wm_img.shape[1], wm_img.shape[0]), flags=cv2.INTER_CUBIC)
rot_bits = extract_bits(rotated, embed_info)
print(f"Recovered from rotation {angle}°:", rot_bits)

# --- Scaling (1.1×) ---
scale_factor = 1.1
scaled = cv2.resize(wm_img, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
# Resize back to original size for extraction
scaled_back = cv2.resize(scaled, (wm_img.shape[1], wm_img.shape[0]), interpolation=cv2.INTER_CUBIC)
scale_bits = extract_bits(scaled_back, embed_info)
print(f"Recovered from scaling {scale_factor}×:", scale_bits)

# --- JPEG Compression (quality 50) ---
jpeg_path = os.path.join(out_dir,"wm_jpeg50.jpg")
cv2.imwrite(jpeg_path, np.uint8(wm_img), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
jpeg_img = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
jpeg_bits = extract_bits(jpeg_img, embed_info)
print("Recovered from JPEG compression (Q50):", jpeg_bits)

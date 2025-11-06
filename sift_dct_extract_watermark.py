import cv2
import numpy as np
import pickle

# --- Settings ---
wm_image_path = r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_simple.png"
embed_info_path = r"C:\Users\91991\Desktop\robust_watermark\outputs\embed_info_simple.pkl"
patch_size = 36
mid_band_coords = [(3,4),(2,3),(4,2),(3,3),(2,4)]
redundancy = 10
half = patch_size // 2

# --- Load watermarked image ---
wm_img = cv2.imread(wm_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# --- Load embedding info ---
with open(embed_info_path,"rb") as f:
    embed_info = pickle.load(f)

# --- Extract bits ---
extracted_bits_all = []

for info in embed_info:
    x, y = info["pt"]
    y0, y1 = y-half, y+half
    x0, x1 = x-half, x+half
    patch = wm_img[y0:y1, x0:x1]

    if patch.shape[0]<8 or patch.shape[1]<8:
        continue

    dct_patch = cv2.dct(patch)
    energy = np.mean([dct_patch[c] for c in mid_band_coords])
    extracted_bits_all.append(1 if energy > 0 else 0)

# --- Majority voting per bit ---
num_bits = len(extracted_bits_all)//redundancy
final_bits = []
for i in range(num_bits):
    votes = extracted_bits_all[i*redundancy:(i+1)*redundancy]
    final_bits.append(1 if votes.count(1)>votes.count(0) else 0)

print("Recovered watermark bits:", final_bits)

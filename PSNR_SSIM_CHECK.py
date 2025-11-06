import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- Paths ---
orig_path = r"C:\Users\91991\Desktop\robust_watermark\data\cameraman.png"
wm_path = r"C:\Users\91991\Desktop\robust_watermark\outputs/watermarked_image.png"

# --- Load images ---
orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)

# --- Ensure same size ---
h, w = orig.shape
wm = cv2.resize(wm, (w,h))

# --- PSNR ---
mse = np.mean((orig.astype(np.float32) - wm.astype(np.float32))**2)
psnr_val = 10 * np.log10(255**2 / mse)
print(f"PSNR: {psnr_val:.2f} dB")

# --- SSIM ---
ssim_val = ssim(orig, wm)
print(f"SSIM: {ssim_val:.4f}")

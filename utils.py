# utils.py
import numpy as np
import pywt
import cv2
from scipy.fftpack import dct, idct
from PIL import Image

# ---------- DWT helpers ----------
def dwt2_gray(img):
    """
    img: 2D numpy array (grayscale, uint8 or float)
    returns LL, LH, HL, HH (floats)
    """
    # ensure float
    arr = np.float32(img)
    coeffs2 = pywt.dwt2(arr, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, LH, HL, HH

def idwt2_from_bands(LL, LH, HL, HH):
    rec = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    return rec

# ---------- Blockwise DCT / IDCT ----------
def blockwise_dct(img_band, block_size=8):
    h, w = img_band.shape
    out = np.zeros_like(img_band, dtype=float)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_band[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                # skip incomplete blocks at edges
                continue
            # 2D DCT via 1D DCT
            tmp = dct(dct(block.T, norm='ortho').T, norm='ortho')
            out[i:i+block_size, j:j+block_size] = tmp
    return out

def blockwise_idct(dct_band, block_size=8):
    h, w = dct_band.shape
    out = np.zeros_like(dct_band, dtype=float)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_band[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                continue
            tmp = idct(idct(block.T, norm='ortho').T, norm='ortho')
            out[i:i+block_size, j:j+block_size] = tmp
    return out

# ---------- Image IO helpers ----------
def load_gray(path):
    img = Image.open(path).convert('L')
    return np.array(img)

def save_gray(path, arr):
    arr2 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr2).save(path)

# ---------- Feature detection ----------
def detect_harris_points(img_gray, max_corners=500, quality=0.01, min_distance=10):
    # img_gray: uint8
    corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=max_corners, qualityLevel=quality,
                                      minDistance=min_distance, blockSize=3, useHarrisDetector=True, k=0.04)
    pts = [(int(x[0][0]), int(x[0][1])) for x in corners] if corners is not None else []
    return pts

def detect_sift_points(img_gray, n_features=500):
    try:
        sift = cv2.SIFT_create(nfeatures=n_features)
    except Exception as e:
        # fallback if SIFT unavailable
        print("SIFT not available in this OpenCV build:", e)
        return []
    kp = sift.detect(img_gray, None)
    pts = [(int(k.pt[0]), int(k.pt[1])) for k in kp]
    return pts

# ---------- utility ----------
def overlay_points_and_save(img_color, points, out_path, radius=3):
    out = img_color.copy()
    for (x,y) in points:
        cv2.circle(out, (x,y), radius, (0,255,0), 1)
    cv2.imwrite(out_path, out)

# ---------- mid-band coords for 8x8 DCT ----------
def mid_band_coords_8():
    # coordinates (row, col) within an 8x8 block to use for embedding
    return [(2,1),(1,2),(2,2),(3,1),(1,3)]

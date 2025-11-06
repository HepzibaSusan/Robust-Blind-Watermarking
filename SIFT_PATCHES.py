import cv2
import numpy as np
import os

# Input image
img = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\data\cameraman.png",
                 cv2.IMREAD_GRAYSCALE)

# Make sure an output folder exists
patch_dir = r"C:\Users\91991\Desktop\robust_watermark\outputs\sift_patches"
os.makedirs(patch_dir, exist_ok=True)

# Detect SIFT keypoints
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)
print("Total keypoints:", len(keypoints))

# Sort by response strength (strongest first)
keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)

# Choose first 5 keypoints for demo
patch_size = 32
half = patch_size // 2
count = 0

for kp in keypoints[:5]:
    x, y = int(kp.pt[0]), int(kp.pt[1])

    # Define patch bounds, clamp to image edges
    x1, x2 = max(0, x - half), min(img.shape[1], x + half)
    y1, y2 = max(0, y - half), min(img.shape[0], y + half)

    patch = img[y1:y2, x1:x2]

    # Save patch
    filename = os.path.join(patch_dir, f"patch_{count}.png")
    cv2.imwrite(filename, patch)
    print(f"Saved {filename}   center=({x},{y})  size={patch.shape}")
    count += 1

print("Done. Check outputs/sift_patches/ for extracted patches.")

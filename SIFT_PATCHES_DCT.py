import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Load a patch (adjust the filename as needed)
patch = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\outputs\sift_patches\patch_0.png",
                   cv2.IMREAD_GRAYSCALE)

# 2. Convert to float32 for DCT
patch_f = np.float32(patch)

# 3. Apply 2-D DCT
dct_patch = cv2.dct(patch_f)

# 4. Visualize (log scale for easier viewing)
plt.imshow(np.log1p(np.abs(dct_patch)), cmap="gray")
plt.title("DCT magnitude of patch_0")
plt.colorbar()
plt.show()

# If you want to inspect or modify a single coefficient, e.g. at row=3, col=4:
print("Original coefficient [3,4]:", dct_patch[3,4])

# Example of embedding a bit (simple demonstration)
bit = 1  # pretend bit to embed
if bit:
    dct_patch[3,4] += 50  # tweak a mid-frequency coefficient

# 5. Inverse DCT to reconstruct the patch
reconstructed = cv2.idct(dct_patch)
reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

# Save to see the difference (will look nearly identical)
cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\patch_0_dct_embed.png",
            reconstructed)
print("Reconstructed patch saved.")

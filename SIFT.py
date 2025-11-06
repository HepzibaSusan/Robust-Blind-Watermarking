import cv2
import numpy as np

# Load grayscale image
img = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\data\cameraman.png",
                 cv2.IMREAD_GRAYSCALE)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(img, None)
print("Number of keypoints detected:", len(keypoints))

# Draw keypoints on a color version for visualization
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_kp = cv2.drawKeypoints(img_color, keypoints, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save result
cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\sift_keypoints.png", img_kp)
print("Saved visualization to outputs/sift_keypoints.png")

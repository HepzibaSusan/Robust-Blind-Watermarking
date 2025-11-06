import cv2

img = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_image.png",
                 cv2.IMREAD_GRAYSCALE)

# Rotate 15 degrees around center
(h, w) = img.shape
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 5, 1.0)  # angle=5°, scale=1.0
rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_rot5.png",
            rotated)

# Scale (resize) to 50%
scaled = cv2.resize(img, (int(w*0.5), int(h*0.5)))
cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_scale50.png",
            scaled)

print("Saved rotated and scaled versions")

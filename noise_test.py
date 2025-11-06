import cv2, numpy as np

img = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_image.png",
                 cv2.IMREAD_GRAYSCALE)
img = np.float32(img)

# Add Gaussian noise (mean 0, std 10)
noise = np.random.normal(0, 10, img.shape)
noisy = img + noise
noisy = np.clip(noisy, 0, 255)

cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_noisy.png",
            np.uint8(noisy))
print("Saved noisy version")

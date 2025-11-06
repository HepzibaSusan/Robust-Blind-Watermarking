import cv2

# Load the watermarked image
img = cv2.imread(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_image.png")

# Save a low-quality JPEG copy (quality = 50 out of 100)
cv2.imwrite(r"C:\Users\91991\Desktop\robust_watermark\outputs\watermarked_jpeg50.jpg",
            img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

print("Saved compressed version at quality 50")

import os
import cv2
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "large_298284_hx7zcry_4qjejncdnmhjysa_r.jpg")
  
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError("Image not found.")


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


b, g, r = cv2.split(img_bgr)


plt.figure(figsize=(8, 6))

# Original RGB image
plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image (RGB) KOSSZIA")
plt.axis("off")

# Red channel
plt.subplot(2, 2, 2)
plt.imshow(r, cmap="gray")
plt.title("RED channel")
plt.axis("off")

# Green channel
plt.subplot(2, 2, 3)
plt.imshow(g, cmap="gray")
plt.title("GREEN channel")
plt.axis("off")

# Blue channel
plt.subplot(2, 2, 4)
plt.imshow(b, cmap="gray")
plt.title("BLUE channel")
plt.axis("off")

plt.tight_layout()
plt.show()

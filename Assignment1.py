import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "large_298284_hx7zcry_4qjejncdnmhjysa_r.jpg")
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError("Image not found.")


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


b, g, r = cv2.split(img_bgr)


def create_color_bars(height, width):
    """Create a color bar test pattern similar to TV test pattern"""
    bars = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = width // 7
    
    # RGB format: [R, G, B]
    colors = [
        [255, 255, 255],  # White
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [0, 255, 0],      # Green
        [255, 0, 255],    # Magenta
        [255, 0, 0],      # Red
        [0, 0, 255]       # Blue
    ]
    
    for i, color in enumerate(colors):
        bars[:, i*bar_width:(i+1)*bar_width] = color
    
    return bars


img_with_bars = img_rgb.copy()
bar_height = img_rgb.shape[0] // 8  
bar_width = img_rgb.shape[1] // 8   
color_bars = create_color_bars(bar_height, bar_width)


y_start = img_rgb.shape[0] - bar_height
x_start = 0
img_with_bars[y_start:y_start+bar_height, x_start:x_start+bar_width] = color_bars


color_bars_bgr = cv2.cvtColor(color_bars, cv2.COLOR_RGB2BGR)


bars_r = color_bars_bgr[:, :, 2].astype(np.float32)  # Red intensity
bars_g = color_bars_bgr[:, :, 1].astype(np.float32)  # Green intensity  
bars_b = color_bars_bgr[:, :, 0].astype(np.float32)  # Blue intensity


bars_r = np.clip(bars_r, 0, 255).astype(np.uint8)
bars_g = np.clip(bars_g, 0, 255).astype(np.uint8)
bars_b = np.clip(bars_b, 0, 255).astype(np.uint8)


r_display = r.copy()
g_display = g.copy()
b_display = b.copy()

r_display[y_start:y_start+bar_height, x_start:x_start+bar_width] = bars_r
g_display[y_start:y_start+bar_height, x_start:x_start+bar_width] = bars_g
b_display[y_start:y_start+bar_height, x_start:x_start+bar_width] = bars_b


plt.figure(figsize=(8, 6))

# Original RGB image with color bars
plt.subplot(2, 2, 1)
plt.imshow(img_with_bars)
plt.title("Original Image (RGB) Kossi_AZ")
plt.axis("off")

# Red channel with color test
plt.subplot(2, 2, 2)
plt.imshow(r_display, cmap="gray", vmin=0, vmax=255)
plt.title("RED channel")
plt.axis("off")

# Green channel with color test
plt.subplot(2, 2, 3)
plt.imshow(g_display, cmap="gray", vmin=0, vmax=255)
plt.title("GREEN channel")
plt.axis("off")

# Blue channel with color test
plt.subplot(2, 2, 4)
plt.imshow(b_display, cmap="gray", vmin=0, vmax=255)
plt.title("BLUE channel")
plt.axis("off")

plt.tight_layout()
plt.show()
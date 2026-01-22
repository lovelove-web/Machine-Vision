import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "rbdk_image.png")
  
img = cv2.imread(image_path)

cv2.circle(img, (695, 583), 55, (0, 0, 255), 6)

# Draw rectangle (image, top_left, bottom_right, color, thickness)
cv2.rectangle(img, (996, 106), (1344, 177), (0, 255, 0), 6)

# Draw circle (image, center, radius, color, thickness)
cv2.rectangle(img, (1106, 542), (1177, 612), (255, 0, 0), 6)

# Put text (image, text, position, font, scale, color, thickness)
cv2.putText(img, 'Kossi AZIADZO NOTATION 23-01-2026', (601, 53), 
cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 6, cv2.LINE_AA)

cv2.putText(img, 'This is Box', (1000, 229), 
cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 100), 3, cv2.LINE_AA)

cv2.putText(img, 'Square 1', (1193, 573), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 2, cv2.LINE_AA)

cv2.putText(img, 'Square in circle', (756, 561), 
cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 100), 2, cv2.LINE_AA)

cv2.imshow('Drawing', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
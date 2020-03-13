import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

flipVertical = cv2.flip(img, 0)
flipHorizontal = cv2.flip(img, 1)
flipBoth = cv2.flip(img, -1)

plt.subplot(221)
plt.imshow(img)
plt.axis("off")

plt.subplot(222)
plt.imshow(flipHorizontal)
plt.axis("off")

plt.subplot(223)
plt.imshow(flipVertical)
plt.axis("off")

plt.subplot(224)
plt.imshow(flipBoth)
plt.axis("off")

plt.show()
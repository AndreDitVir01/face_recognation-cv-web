import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('image_noise.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#-=Filtering =-
## 2D Convolution

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(521),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(522),plt.imshow(dst),plt.title('2D Convolution')
plt.xticks([]), plt.yticks([])

## Image Blurring (Image Smoothing)
### Averaging

blur = cv2.blur(img,(5,5))

plt.subplot(523),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(524),plt.imshow(blur),plt.title('Blurred Averaging')
plt.xticks([]), plt.yticks([])

### Gaussian Filtering

blur = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(525),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(526),plt.imshow(blur),plt.title('Blurred Gaussian')
plt.xticks([]), plt.yticks([])

### Median Filtering

median = cv2.medianBlur(img,5)

plt.subplot(527),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(528),plt.imshow(blur),plt.title(' Median')
plt.xticks([]), plt.yticks([])

### Bilateral Filtering

blur = cv2.bilateralFilter(img,9,75,75)

plt.subplot(529),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(5,2,10),plt.imshow(blur),plt.title('Blurred Bilateral')
plt.xticks([]), plt.yticks([])


## Show Plot
plt.show()
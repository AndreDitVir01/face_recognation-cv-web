# import cv2 
# import numpy as np  

# cap = cv2.imread('image.jpg')
# #You're free to do a resize or not, just for the example
# cap = cv2.resize(cap, (340,480))
# for x in range (0,340,1):
#     for y in range(0,480,1):
#         color = cap[y,x]
#         print(color)

import cv2
import numpy as np

image1 = cv2.imread('image.jpg')

rows = image1.shape[0]
cols = image1.shape[1]

image2 = np.zeros((rows,cols,3), np.uint8)

for i in range(rows):
    for j in range(cols):
    	b,g,r=image1[i,j]
    	image2[i,j]=[0,g,0]

cv2.imshow('Original Image', image1)
cv2.imshow('Green Image', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
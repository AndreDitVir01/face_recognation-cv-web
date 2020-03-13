import cv2 # Not actually necessary if you just want to create an image.
import numpy as np
blank_image = np.zeros((500,500,3), np.uint8)
blank_image[0:500//2,:] = (0,0,255)      # (B, G, R)
blank_image[500//2:500,:] = (255,255,255)
cv2.imshow('img', blank_image)
cv2.waitKey(0)
cv2.destroyWindow()
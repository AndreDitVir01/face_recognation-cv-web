import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

#DETECT THEM!

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('wajah_target.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv2.imshow('img',img)
print("Found {0} Faces!".format(len(faces)))
sub = 520
i=1

# SAVE THEM!
for (x, y, w, h) in faces:
    sub +=1
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = img[y:y + h, x:x + w]
    
    ## PLOT THEM!
    plt.subplot(sub)
    plt.imshow(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))

    plt.title("Object ke -" + str(i)), plt.xticks([]), plt.yticks([])

    ## HIST THEM!
    sub+=1
    plt.subplot(sub)
    plt.hist(roi_color.ravel(),256,[0,256])

    print("[INFO] Object found. Saving locally.")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

    i+=1

plt.show()
status = cv2.imwrite('faces_detected.jpg', img)
print("[INFO] img faces_detected.jpg written to filesystem: ", status)

## SHOW THEM!
cv2.waitKey(0)
cv2.destroyAllWindows()
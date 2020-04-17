import numpy as np
import cv2
import math
import os
from matplotlib import pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))

#DETECT THEM!

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

directory = os.path.join(script_path, "testingall")
listds = os.listdir(directory)
number_files = len(listds)
os.chdir(directory)
arr = os.listdir()

for i in range(number_files):
    directory = os.path.join(script_path, "testingall")
    os.chdir(directory)
    image_name = arr[i]
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

    j=0

    # SAVE THEM!
    for (x, y, w, h) in faces:

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]

        # plt.title("Object ke -" + str(i)), plt.xticks([]), plt.yticks([])

        # print("[INFO] Object found. Saving locally.")

        directory = r'D:\Teknik Informatika\CV\citra02\citra02\testing'
        os.chdir(directory)
        nama = str(j)+image_name
        cv2.imwrite(nama, roi_color)

        j+=1
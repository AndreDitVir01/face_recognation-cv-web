import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.feature import local_binary_pattern
import math
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

METHOD = 'uniform'

# settings for LBP
radius = 2
n_points = 2 * radius
dataset = {}

########################################
#membaca dataset dan menyimpan ke dictionary

directory =r"D:\Teknik Informatika\CV\citra02\citra02\Dataset"
listds = os.listdir(directory)
number_files = len(listds)

os.chdir(directory)
arr = os.listdir()


for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	i=0
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_color = image[y:y + h, x:x + w]
		print("[INFO] Object found. Saving locally.")
		cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
		i+=1

	lbp = local_binary_pattern(roi_color, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))
	
	j = i+31
	nrp = '21031810' + str(j)

	dataset.update({nrp : hist})

###############################################
#membandingkan fitur

def match(counts):
	best_score = 10
	best_name = None
	for key,value in dataset.items():
		score = 0
		for o in range(6):
			score+=abs(value[o]-counts[o])
		if score < best_score:
			best_score = score
			best_name = key
	return best_name


directory = r"D:\Teknik Informatika\CV\citra02\citra02\Testing01"
listds = os.listdir(directory)
number_files = len(listds)
os.chdir(directory)
arr = os.listdir()

for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))

	j = i+31
	nrp = '21031810' + str(j)

	print('gambar : ' + image_name + '/nrp : ' + nrp)
	print('result : ' + match(hist))
	print()
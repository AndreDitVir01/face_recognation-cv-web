import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.feature import local_binary_pattern

METHOD = 'uniform'

# settings for LBP
radius = 2
n_points = 2 * radius
dataset = {}

########################################
#membaca dataset dan menyimpan ke dictionary

directory = r'E:\py\citra03\dataset'
listds = os.listdir(directory)
number_files = len(listds)

os.chdir(directory)
arr = os.listdir()

for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))

	nrp = '0000' + str(i)

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


directory = r'E:\py\citra03\testing'
listds = os.listdir(directory)
number_files = len(listds)
os.chdir(directory)
arr = os.listdir()

for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))

	nrp = 'T000' + str(i)

	print('gambar : ' + image_name + '/nrp : ' + nrp)
	print('result : ' + match(hist))
	print()
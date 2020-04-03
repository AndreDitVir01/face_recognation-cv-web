import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.feature import local_binary_pattern

METHOD = 'uniform'
script_path = os.path.dirname(os.path.realpath(__file__))

# settings for LBP
radius = 2
n_points = 2 * radius
dataset = {}

########################################
#membaca dataset dan menyimpan ke dictionary

maps_path = os.path.join(script_path, "dataset")
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()

for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))

	nrp = '0000' + str(i)

	dataset.update({image_name : hist})

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

maps_path = os.path.join(script_path, "testing01")
listds = os.listdir(maps_path)
number_files = len(listds)
os.chdir(maps_path)
arr = os.listdir()

for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))

	if(i==26):
		nrp = 2103181060
	elif(i>=10):
		nrp = 2103181032 + i
	else :
		nrp = 2103181031 + i

	print('gambar : ' + image_name + '/nrp : ' + str(nrp))
	print('result : ' + match(hist))
	print()

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

maps_path = r"D:\Teknik Informatika\CV\citra02\citra02\dataset"
# maps_path = os.path.join(script_path, "dataset")
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()

j = 31
for i in range(number_files):
	
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))


	if(j==41):
		j+=1
	elif(j==58):
		j+=2

	nrp = '21031810' + str(j)

	dataset.update({nrp : hist})
	j+=1

j=31

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


# maps_path = os.path.join(script_path, "testing01")
maps_path = r"D:\Teknik Informatika\CV\citra02\citra02\testing02"
listds = os.listdir(maps_path)
number_files = len(listds)
os.chdir(maps_path)
arr = os.listdir()

counterOfStatus = 0
for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	lbp = local_binary_pattern(image, n_points, radius, METHOD)

	hist, _ = np.histogram(lbp, density=True, bins=6, range=(0,6))

	if(j==41):
		j+=1
	elif(j==58):
		j+=2

	nrp = '21031810' + str(j)

	status = "Keluaran Tidak Cocok"
	if(nrp == match(hist)):
		status = "Keluaran Cocok"
		counterOfStatus+=1

	print('Gambar : ' + image_name + '/NRP : ' + nrp)
	print('Hasil : ' + match(hist) + '/Status : '+status)
	print()
	j+=1

print('Hasil yang cocok = '+ str( 27 - counterOfStatus))
print('Hasil yang tidak cocok = '+ str(counterOfStatus))
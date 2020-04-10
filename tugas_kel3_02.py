import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.feature import local_binary_pattern

METHOD = 'uniform'
script_path = os.path.dirname(os.path.realpath(__file__))

# settings for LBP
radius = 3
n_points = 8 * radius
dataset = {}

###############################################
# Membandingkan fitur

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

########################################
# Membaca DATASET dan menyimpan ke dictionary
####################################################
# Dataset
maps_path = "../citra02/dataset"
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()

# Menampilkan nama path DATASET
print("Dataset = " + maps_path)
print("Number files dataset = " + str(number_files))

# Looping untuk DATASET
j = 31
printBins = False
for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	
	## Ekstraksi foto menjadi LBP
	lbp = local_binary_pattern(image, n_points, radius, METHOD)
	n_bins = int(lbp.max() + 1)
	
	## Cetak Number Bins satu kali saja
	if(printBins== False):
		print("Number bins = " + str(n_bins))
		printBins = True

	hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0,n_bins))

	### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
	if(j==41):
		j+=1
	elif(j==58):
		j+=2

	### Menyimpan dengan key NRP
	nrp = '21031810' + str(j)
	### Menyimpan hist ke key NRP
	dataset.update({nrp : hist})

	## Cetak nama file gambar lengkap beserta pathnya "untuk menunjukkan telah tersimpan"
	print(maps_path + "/" + image_name + "::" + str(nrp))
	## Increment nilai J
	j+=1

####################################################
# Testing01
# Batasi tampilan dan reset counter j
print('============================================================')
j=31

maps_path = r"D:\Teknik Informatika\CV\citra02\citra02\testing01"
listds = os.listdir(maps_path)
number_files = len(listds)
os.chdir(maps_path)
arr = os.listdir()

# Menampilkan nama path DATASET
print("Testing = " + maps_path)
print("Number files dataset = " + str(number_files))

counterOfStatus = 0
for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	## Ekstraksi foto menjadi LBP
	lbp = local_binary_pattern(image, n_points, radius, METHOD)
	n_bins = int(lbp.max() + 1)

	hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0,n_bins))

	### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
	if(j==41):
		j+=1
	elif(j==58):
		j+=2

	nrp = '21031810' + str(j)

	status = "0"
	if(nrp == match(hist)):
		status = "1"
		counterOfStatus+=1

	print(nrp + ' ::best answer = ' + match(hist) + ' ' + status)
	j+=1

print('Hasil yang cocok = '+ str(counterOfStatus))
print('Hasil yang tidak cocok = '+ str(27 - counterOfStatus))
print('Akurasi = ' + str((counterOfStatus/number_files)*100) + '%')

####################################################
# Testing02
# Batasi tampilan dan reset counter j
print('============================================================')
j=31

maps_path = r"D:\Teknik Informatika\CV\citra02\citra02\testing02"
listds = os.listdir(maps_path)
number_files = len(listds)
os.chdir(maps_path)
arr = os.listdir()

# Menampilkan nama path DATASET
print("Testing = " + maps_path)
print("Number files dataset = " + str(number_files))

counterOfStatus = 0
for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)
	## Ekstraksi foto menjadi LBP
	lbp = local_binary_pattern(image, n_points, radius, METHOD)
	n_bins = int(lbp.max() + 1)

	hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0,n_bins))

	### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
	if(j==41):
		j+=1
	elif(j==58):
		j+=2

	nrp = '21031810' + str(j)

	status = "0"
	if(nrp == match(hist)):
		status = "1"
		counterOfStatus+=1

	print(nrp + ' ::best answer = ' + match(hist) + ' ' + status)
	j+=1

print('Hasil yang cocok = '+ str(counterOfStatus))
print('Hasil yang tidak cocok = '+ str(27 - counterOfStatus))
print('Akurasi = ' + str((counterOfStatus/number_files)*100) + '%')
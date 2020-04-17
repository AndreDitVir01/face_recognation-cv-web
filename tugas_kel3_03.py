# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
import os
from skimage.feature import local_binary_pattern

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

# DETEKSI WAJAH
## Arahkan ke folder testing foto
maps_path = r"D:\Teknik Informatika\CV\citra02\citra02\TestingAll"
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()

## Mendeteksi wajah dengan haarcascades
face_cascade = cv2.CascadeClassifier('D:\\Teknik Informatika\\CV\\citra02\\citra02\\haarcascade_frontalface_default.xml')
c = 31
for k in range(number_files):
	image_name = arr[k]
	image = cv2.imread(image_name)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		### Pemberian kotak biru untuk penanda
		img = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
	print("Ditemukan {0} wajah!".format(len(faces)))
    # SAVE THEM!
	for (x, y, w, h) in faces:
		### Pemberian kotak hijau untuk penanda simpan
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_color = img[y:y + h, x:x + w]
		print("[INFO] Object found. Saving locally.")
		cv2.imwrite('D:\\Teknik Informatika\\CV\\citra02\\citra02\\TempTestArea\\21031810' + str(c) +'_face.jpg', roi_color)
		c+=1

# EKSTRAKSI FITUR
METHOD = 'uniform'
script_path = os.path.dirname(os.path.realpath(__file__))

# settings for LBP
radius = 2
n_points = 2 * radius
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
maps_path = 'D:\\Teknik Informatika\\CV\\citra02\\citra02\\Dataset'
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
	print(maps_path + "\\" + image_name + "::" + str(nrp))
	## Increment nilai J
	j+=1

####################################################
# Testing01
# Batasi tampilan dan reset counter j
print('============================================================')
j=31

maps_path = 'D:\\Teknik Informatika\\CV\\citra02\\citra02\\TempTestArea'
listds = os.listdir(maps_path)
number_files = len(listds)
os.chdir(maps_path)
arr = os.listdir()

# Menampilkan nama path DATASET
print("Testing = " + maps_path)
print("Number files testing = " + str(number_files))

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
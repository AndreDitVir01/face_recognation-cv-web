# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
import os
import array
from skimage.feature import local_binary_pattern

import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

##########
# DETEKSI WAJAH
##########

# EKSTRAKSI FITUR
METHOD = 'uniform'
script_path = os.path.dirname(os.path.realpath(__file__))

# settings for LBP
radius = 2
n_points = 2 * radius
dataset = {}
testing = {}

########## 1) MEMBACA FOLDER DATASET

# Menampilkan nama path DATASET
maps_path = 'D:\\Teknik Informatika\\CV\\citra02\\citra02\\Dataset'
listds = os.listdir(maps_path)
number_files = len(listds)
os.chdir(maps_path)
arr = os.listdir()

print("Dataset = " + maps_path)
print("Number files dataset = " + str(number_files))
# Looping untuk DATASET
j = 31
printBins = False
for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)

	### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
	if(j==41):
		j+=1
	elif(j==58):
		j+=2
	nrp = '21031810' + str(j)

	## Cetak nama file gambar lengkap beserta pathnya "untuk menunjukkan telah terbaca"
	print(maps_path + "\\" + image_name + "::" + str(nrp))
	## Increment nilai J
	j+=1

########## 2) DETEKSI WAJAH DENGAN HAARCASCADES DAN SIMPAN DI FOLDER DATASET_HAAR HASIL DETEKSI WAJAH DALAM BENTUK GRAY

## Arahkan ke folder dataset foto
maps_path = r"D:\Teknik Informatika\CV\citra02\citra02\Dataset"
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
	# print("Ditemukan {0} wajah!".format(len(faces)))
    # SAVE THEM!
	for (x, y, w, h) in faces:
		### Pemberian kotak hijau untuk penanda simpan
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_color = img[y:y + h, x:x + w]
		roi_gray = gray[y:y+h, x:x+w]	
		### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
		if(c==41):
			c+=1
		elif(c==58):
			c+=2
		# print("[INFO] Object found. Saving locally.")
		cv2.imwrite('D:\\Teknik Informatika\\CV\\citra02\\citra02\\Dataset_Haar\\21031810' + str(c) +'_face.jpg', roi_gray)
		c+=1

########## 3) LAKUKAN EKSTRAKSI FITUR FILE YANG DI DATASET_HAAR DAN SIMPAN DI DATABASE FITUR
j = 31
# Dataset_Haar	
maps_path = 'D:\\Teknik Informatika\\CV\\citra02\\citra02\\Dataset_Haar'
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()

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

	## Increment nilai J
	j+=1

########## 4) UNTUK TESTING BACA FOLDER TESTING

# Testing	
maps_path = 'D:\\Teknik Informatika\\CV\\citra02\\citra02\\Testing01'
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()

print("Testing = " + maps_path)
print("Number files testing = " + str(number_files))
# Looping untuk TESTING
j = 31
printBins = False
for i in range(number_files):
	image_name = arr[i]
	image = cv2.imread(image_name, 0)

	### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
	if(j==41):
		j+=1
	elif(j==58):
		j+=2

	nrp = '21031810' + str(j)
	## Cetak nama file gambar lengkap beserta pathnya "untuk menunjukkan telah terbaca"
	print(maps_path + "\\" + image_name + "::" + str(nrp))
	## Increment nilai J
	j+=1

########## 5) DETEKSI WAJAH DI TESTING DENGAN HAARCASCADES DAN SIMPAN DI FOLDER TESTING_HAAR

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
	# print("Ditemukan {0} wajah!".format(len(faces)))
    # SAVE THEM!
	for (x, y, w, h) in faces:
		### Pemberian kotak hijau untuk penanda simpan
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi_color = img[y:y + h, x:x + w]
		roi_gray = gray[y:y+h, x:x+w]	
		### NRP 41 dan 58, 59 dilompati karena tidak ada mahasiswa dengan nrp tersebut
		if(c==41):
			c+=1
		elif(c==58):
			c+=2
		# print("[INFO] Object found. Saving locally.")
		cv2.imwrite('D:\\Teknik Informatika\\CV\\citra02\\citra02\\Testing_Haar\\21031810' + str(c) +'_face.jpg', roi_gray)
		c+=1

########## 6) LAKUKAN EKSTRAKSI FITUR FILE YANG DI TESTING_HAAR

# Testing_Haar	
maps_path = 'D:\\Teknik Informatika\\CV\\citra02\\citra02\\Testing_Haar'
listds = os.listdir(maps_path)
number_files = len(listds)

os.chdir(maps_path)
arr = os.listdir()
j = 31
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
	testing.update({nrp : hist})

	## Increment nilai J
	j+=1

########## 7) NILAI FITUR UNTUK SETIAP FILE HARUS DILAKUKAN NORMALISASI TERLEBIH DAHULU, DENGAN CARA : NILAI FITUR KE - N (NILAI  FITUR BIN KE N/NILAI MAX FITUR PADA FILE TSB)* 100%

# fitur dataset dari dataset_haar
for key,value in dataset.items():
	new_value = array.array('f', [])
	for o in range(6):
		new_value.append((value[o]/max(value))*100)
	dataset.update({key : new_value})

# fitur testing dari testing_haar
for key2,value2 in testing.items():
	new_value2 = array.array('f', [])
	for o in range(6):
		new_value2.append((value2[o]/max(value2))*100)
	testing.update({key2 : new_value2})

########## 8) FITUR WAJAH TESTING DIBANDINGKAN DENGAN DATABASE FITUR DAN CARI NILAI JARAK YANG PALING KECIL SEBAGAI JAWABANNYA

# perbandingan nilai fitur

counterOfStatus=0
for key,value in testing.items():
	z=0
	best_name = None
	y=0
	temp_score = 700
	for key2,value2 in dataset.items():
		selisih_perMahasiswa = array.array('f', [])
		total_perMahasiswa=0
		for o in range(6):
			selisih_perMahasiswa.append(abs(value[o]-value2[o]))
		for p in range(6):
			total_perMahasiswa+=selisih_perMahasiswa[p]
		if(total_perMahasiswa<temp_score):
			temp_score = total_perMahasiswa
			best_name = key2
		y+=1
		z+=1
	status = "0"
	if(key == best_name):
		status = "1"
		counterOfStatus+=1
	print(key + '::best answer = ' + best_name + ' ::status = ' + status)

# CETAK NAMA TERBAIK

print('Hasil yang cocok = '+ str(counterOfStatus))
print('Hasil yang tidak cocok = '+ str(27 - counterOfStatus))
print('Akurasi = ' + str((counterOfStatus/number_files)*100) + '%')
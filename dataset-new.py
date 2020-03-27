"""
===============================================
Local Binary Pattern for texture classification
===============================================

In this example, we will see how to classify textures based on LBP (Local
Binary Pattern). LBP looks at points surrounding a central point and tests
whether the surrounding points are greater than or less than the central point
(i.e. gives a binary result).

Before trying out LBP on an image, it helps to look at a schematic of LBPs.
The below code is just used to plot the schematic.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

###

# Opening a file  
file1 = open('myfile.txt', 'w')  
    
# Writing a string to file  
# file1.write(s)
  

# A function that calls 2 functions to separately  
# listing out directories and files. 
# It takes a default argument as cwd(.). We can  
# pass other paths too. 

import os 

path=".\dataset"
f_list = [] 
    
try: 
    for f in os.listdir(path): 
        if os.path.isfile(os.path.join(path, f)): 
            f_list.append(f)       
except: 
    print ("\nError, once check the path")

######

METHOD = 'uniform'
plt.rcParams['font.size'] = 9

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from skimage.feature import greycomatrix

# settings for LBP
radius = 2
n_points = 2 * radius


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

def hist(ax, lbp):
        n_bins = int(lbp.max() + 1)
        return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                    facecolor='0.5')

dataset = []
j=0
for i in range(len(f_list)-1):
    image = cv2.imread(".\dataset\\"+f_list[i], 0)
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    # plot histograms of LBP of textures
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()
    titles = ('edge', 'flat', 'corner')
    w = width = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4            # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)      # 3 /4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                    list(range(i_34 - w, i_34 + w + 1)))
    label_sets = (edge_labels, flat_labels, corner_labels)
    print('n_bins = ', n_bins)
    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
    dataset[j] = counts
    j+=1

######

# s = path.split('\\')[1]
# for i in range(len(f_list)-1):
#     file1.write(s+"/"+f_list[i]+"\t nrp = "+f_list[i].split('.')[0]+"\n")
#     file1.write(str(counts2)+"\n\n")

# # Writing multiple strings  
# # at a time  
# # file1.writelines("\n"+str(counts2)) 

# # Closing file  
# file1.close()  
    
# # Checking if the data is  
# # written to file or not  
# file1 = open('myfile.txt', 'r')  

# print(file1.read())  
# file1.close()  
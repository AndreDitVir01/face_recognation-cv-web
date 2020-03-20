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


image = cv2.imread('foto1.jpg', 0)
image2 = cv2.imread('foto2.jpg', 0)
image3 = cv2.imread('foto2.jpg', 0)
image3 = cv2.flip(image3, 0)

lbp = local_binary_pattern(image, n_points, radius, METHOD)
lbp2 = local_binary_pattern(image2, n_points, radius, METHOD)
lbp3 = local_binary_pattern(image3, n_points, radius, METHOD)

n_bins = int(lbp.max() + 1)

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
fig2, (ax_img2, ax_hist2) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
fig3, (ax_img3, ax_hist3) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))

plt.gray()

titles = ('edge', 'flat', 'corner')

w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(image, lbp, labels))

for ax2, labels in zip(ax_img2, label_sets):
    ax2.imshow(overlay_labels(image2, lbp2, labels))

for ax3, labels in zip(ax_img3, label_sets):
    ax3.imshow(overlay_labels(image3, lbp3, labels))

print('n_bins = ', n_bins)
for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

# print(counts) # Print counts of feature
for i in range(n_bins):
    print ('%.8f'%counts[i])
ax_hist[0].set_ylabel('Percentage')

###

print('n_bins = ', n_bins)
for ax, labels, name in zip(ax_hist2, label_sets, titles):
    counts2, _, bars = hist(ax, lbp2)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

# print(counts2) # Print counts of feature
for i in range(n_bins):
    print ('%.8f'%counts2[i])
ax_hist2[0].set_ylabel('Percentage')

###

print('n_bins = ', n_bins)
for ax, labels, name in zip(ax_hist3, label_sets, titles):
    counts3, _, bars = hist(ax, lbp3)
    highlight_bars(bars, labels)
    ax.set_ylim(top=np.max(counts[:-1]))
    ax.set_xlim(right=n_points + 2)
    ax.set_title(name)

# print(counts2) # Print counts of feature
for i in range(n_bins):
    print ('%.8f'%counts3[i])
ax_hist2[0].set_ylabel('Percentage')

###

for ax in ax_img:
    ax.axis('off')

for ax in ax_img2:
    ax.axis('off')

for ax in ax_img3:
    ax.axis('off')

plt.show()

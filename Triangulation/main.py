import os
import numpy as np
import cv2 as cv
import funcs

import matplotlib.pyplot as plt

from sfm import Sfm


folder = 'datasets/pig'
cameras = [np.loadtxt(folder + '/calib/' + file) for file in sorted(os.listdir(folder + '/calib'))]

sil = [255 - cv.imread(folder + '/silhouettes/' + file) for file in sorted(os.listdir(folder + '/silhouettes')) if file[4:] == '.pgm']
imgs = [cv.imread(folder + '/images/' + file) for file in sorted(os.listdir(folder + '/images')) if file[4:] in ['.jpg', '.ppm']]
images = [image * s for image, s in zip(imgs, sil)]

sfm = Sfm(images, cameras)
cloud = sfm.build()
cloud[:,2] *= -1
funcs.plot_cloud(cloud)

# plt.imshow(cv.cvtColor(imgs[3], cv.COLOR_BGR2RGB))
# plt.imshow(images[3])
# plt.show()
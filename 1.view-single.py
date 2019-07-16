#!/usr/bin/python3
import numpy as np
import time
import cv2

from src.settings import *
from src.spectral_image import *


if True:
    path = '/media/ovan/684C-0692/'
    folder = '20190423_053725'
    prefix = '0008_'
    height = '2.0'
else:
    path = './data/'
    folder = 'steep/5.0'
    prefix = ''
    height = '5.0'
pass

start_time = time.time()
S = SpectralImage(path, folder, prefix, height)
    
loaded, nb_kp = load_spectral_bands(
    S, method='AKAZE',
    reference=all_bands.index(710),
    verbose=3
)

text = f"--- {time.time() - start_time} seconds ---"
print('-' * len(text))
print(text)
print('-' * len(text))

images = [i/i.max()*255 for i in loaded]
images = np.array(images).std(axis=0)*4
images = images.astype('uint8')
images = cv2.applyColorMap(images, cv2.COLORMAP_JET)

false_color = compute_false_color(loaded)

cv2.imshow('mean', images)
cv2.imshow('false color', false_color)
cv2.waitKey(1)

while cv2.waitKey() != 27:
    pass
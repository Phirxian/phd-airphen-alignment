#!/usr/bin/python3
import numpy as np
import time
import cv2

from airphen.spectral_image import *

if True:
    path = '/media/ovan/31F66DD04B004F4B/database/inra2/'
    folder = '20190416_055537'
    prefix = '0008_'
    height = './data/2.0.npy'
else:
    path = './data/'
    folder = 'step/1.6'
    prefix = ''
    height = './data/1.6.npy'
pass

start_time = time.time()
S = SpectralImage(path, folder, prefix, height)
    
loaded = S.loaded
loaded, nb_kp = S.spectral_registration(
    method='GFTT1',
    reference=all_bands.index(570),
    verbose=3
)

text = f"--- {time.time() - start_time} seconds ---"
print('-' * len(text))
print(text)
print('-' * len(text))

false_color = S.compute_false_color()
cv2.imshow('false color', false_color)
cv2.waitKey(1)

while cv2.waitKey() != 27:
    pass
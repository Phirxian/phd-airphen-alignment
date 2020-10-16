#!/usr/bin/python3
import numpy as np
import time
import cv2

from airphen.spectral_image import *

if True:
    path = '/media/ovan/31F66DD04B004F4B1/database/inra2/'
    folder = 'haricot1/'
    prefix = '0008_'
    height = '2.0.npy'
else:
    path = '/home/javayss/Documents/code/merge/data/'
    folder = ''
    prefix = '1060_'
    height = './data/1.8.npy'
pass

start_time = time.time()
S = SpectralImage(path, folder, prefix, 'data/', height)
    
loaded = S.loaded
loaded, nb_kp = S.spectral_registration(
    method='GFTT',
    reference=all_bands.index(570),
    verbose=1
)

text = f"--- {time.time() - start_time} seconds ---"
print('-' * len(text))
print(text)
print('-' * len(text))

false_color = S.compute_false_color()
cv2.imshow('false color', false_color)
cv2.waitKey(1)

for i,b in enumerate(loaded):
    b = gradient_normalize(b, 0.1)
    cv2.imwrite('/tmp/bands-'+str(i)+'.png', b.astype('uint8'))
    b = build_gradient(b, method='Ridge')
    cv2.imwrite('/tmp/grad-'+str(i)+'.png', b)

while cv2.waitKey() != 27:
    pass
#!/usr/bin/python3
import numpy as np
import time
import cv2

from airphen.data import *
from airphen.spectral_image import *

for h in height:
    start_time = time.time()
    S = SpectralImage(
        './data/steep/', str(h), '',
        h # use linear model
        #'./data/' + str(h) + '.npy'
    )
    
    loaded, nb_kp = S.spectral_registration(
        verbose=1, method='GFTT',
        reference=all_bands.index(570)
    )
    
    print('-------------------')
    print(h, 'loaded in ', time.time() - start_time)
    print('-------------------')
    
    images = np.array([i/i.max()*255 for i in loaded])
    mean = images.mean(axis=0)
    std = images.std(axis=0)*4
    mean = cv2.applyColorMap(mean.astype('uint8'), cv2.COLORMAP_JET)
    std = cv2.applyColorMap(std.astype('uint8'), cv2.COLORMAP_JET)
    
    false_color = S.compute_false_color()
        
    cv2.imshow('std', mean)
    cv2.imshow('mean', std)
    cv2.imshow('false color', false_color)
    #cv2.imwrite('results/' + str(h) + '_mean' + '.jpg', mean)
    #cv2.imwrite('results/' + str(h) + '_std' + '.jpg', std)
    #cv2.imwrite('results/' + str(h) + '_false_color' + '.jpg', false_color)
    cv2.waitKey(1)
    
    #while cv2.waitKey() != 27:
    #    pass
pass
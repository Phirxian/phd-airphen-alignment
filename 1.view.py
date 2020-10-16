#!/usr/bin/python3
import numpy as np
import time
import cv2

from airphen.data import *
from airphen.spectral_image import *
from timeit import default_timer as timer

for h in height:
    start_time = time.time()
    S = SpectralImage(
        #'./data/steep-chess/', str(h), '',
        './data/step/', str(h), '', 'data/',
        h, # use linear model
        #'./data/' + str(h) + '.npy'
    )
    
    start = timer()
    loaded, nb_kp = S.spectral_registration(
        verbose=1, method='GFTT1',
        #reference=all_bands.index(570)
        reference=None#-(5+1)
    )
    print('registration time', timer()-start)
    
    print('-------------------')
    print(h, 'loaded in ', time.time() - start_time)
    print('-------------------')
    
    images = np.array([i/i.max()*255 for i in loaded])
    mean = images.mean(axis=0)
    std = images.std(axis=0)*4
    mean = cv2.applyColorMap(mean.astype('uint8'), cv2.COLORMAP_JET)
    std = cv2.applyColorMap(std.astype('uint8'), cv2.COLORMAP_JET)
    
    false_color = S.compute_false_color()
    
    for i,b in enumerate(loaded):
        b = false_color_normalize(b) * 255
        b = b.astype('uint8')
        cv2.imwrite('/tmp/bands-'+str(i)+'.png', b)
        
        
    cv2.imshow('std', mean)
    cv2.imshow('mean', std)
    cv2.imshow('false color', false_color)
    #cv2.imwrite('results/' + str(h) + '_mean' + '.jpg', mean)
    #cv2.imwrite('results/' + str(h) + '_std' + '.jpg', std)
    #cv2.imwrite('results/' + str(h) + '_false_color' + '.jpg', false_color)
    cv2.waitKey(1)
    
    while cv2.waitKey() != 27:
        pass
pass
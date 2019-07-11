#!/usr/bin/python3
import numpy as np
import cv2

from src.settings import *
from src.spectral_image import *

height = [
    '1.6', '1.8', '2.0',
    '2.2', '2.4', '2.6', '2.8', '3.0',
    '3.2', '3.4', '3.6', '3.8', '4.0',
    '4.2', '4.4', '4.6', '4.8', '5.0'
]

for h in height:
    x = SpectralImage()
    load_spectral_bands(x, './data/steep/', h)
    
    print('-------------------')
    print(h, 'loaded')
    print('-------------------')
    
    images = []
    
    for it in x.images.items():
        img = it[1]
        img = img - img.min()
        img = img/img.max()*255
        images.append(img)
    pass

    stereo = cv2.StereoBM_create()
    stereo.setPreFilterSize(9)     # 63
    stereo.setPreFilterCap(31)      # 11
    stereo.setBlockSize(15)         # 39
    stereo.setMinDisparity(0)       # 0
    stereo.setNumDisparities(16)    # 80
    stereo.setTextureThreshold(10)  # 16
    stereo.setUniquenessRatio(15)    # 5
    stereo.setSpeckleWindowSize(100) # 41
    stereo.setSpeckleRange(4)      # 11
    disparity = stereo.compute(images[0].astype('uint8'),images[2].astype('uint8'))
    disparity = disparity/disparity.max()*255
    disparity = disparity.astype('uint8')
    
    images = np.array(images).mean(axis=0).astype('uint8')
    images = cv2.applyColorMap(images, cv2.COLORMAP_JET)
        
    cv2.imshow('mean', images)
    cv2.imshow('disparity', disparity)
    cv2.imshow('false color', x.false_color)
    cv2.imwrite('results/' + h + '_mean' + '.jpg', images)
    cv2.imwrite('results/' + h + '_disparity' + '.jpg', disparity)
    cv2.imwrite('results/' + h + '_false_color' + '.jpg', x.false_color)
    cv2.waitKey(1)
    
    #while cv2.waitKey() != 27:
    #    pass
pass
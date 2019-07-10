#!/usr/bin/python3
import numpy as np
import rasterio
import cv2
import os

from settings import *

class SpectralImage:
    def __init__(self):
        pass
    pass
    
    def read_tiff(self, fname):
        if not os.path.exists(fname): return None
        # skep detection if the corresponding output exist
        #if os.path.exists(csv): continue
        
        geotiff = rasterio.open(fname)
        data = geotiff.read()
        tr = geotiff.transform
        
        return data[0]
    pass
pass

height = [
    '1.6', '1.8', '2.0',
    '2.2', '2.4', '2.6', '2.8', '3.0',
    '3.2', '3.4', '3.6', '3.8', '4.0',
    '4.2', '4.4', '4.6', '4.8', '5.0'
]

path = '/media/ovan/6d1bbc2c-2d2d-4126-960f-d57d6de8ae10/'

for h in height:
    x = SpectralImage()
    load_spectral_bands(x, path+'portique/steep/', h)
    
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
    
    images = np.array(images).mean(axis=0).astype('uint8')
    images = cv2.applyColorMap(images, cv2.COLORMAP_JET)
        
    cv2.imshow('mean', images)
    cv2.imshow('false color', x.false_color)
    cv2.imwrite('results/' + h + '_mean' + '.jpg', images)
    cv2.imwrite('results/' + h + '_false_color' + '.jpg', x.false_color)
    cv2.waitKey(1)
    
    #while cv2.waitKey() != 27:
    #    pass
pass
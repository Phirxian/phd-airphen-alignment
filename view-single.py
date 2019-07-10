#!/usr/bin/python3
import numpy as np
import rasterio
import cv2
import os
import time

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

x = SpectralImage()
path = '/media/ovan/684C-0692/'

start_time = time.time()
load_spectral_bands(x, path, '20190416_055537', '0008_', '2.0')

text = f"--- {time.time() - start_time} seconds ---"
print('-' * len(text))
print(text)
print('-' * len(text))

images = []

for it in x.images.items():
    img = it[1]
    img = img/img.max()*255
    images.append(img)
pass

images = np.array(images).mean(axis=0).astype('uint8')
images = cv2.applyColorMap(images, cv2.COLORMAP_JET)
    
cv2.imshow('mean', images)
cv2.imshow('false color', x.false_color)
cv2.waitKey(1)

while cv2.waitKey() != 27:
    pass
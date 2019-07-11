#!/usr/bin/python3
import numpy as np
import rasterio
import cv2
import os
import time

from src.settings import *

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
    #img = img.reshape([*img.shape, 1])
    #img = img[:820,:1220,:]
    #print(img.shape)
    images.append(img)
    print(it[0])
pass

if True:
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
else:
    stereo = cv2.StereoSGBM_create()
    stereo.setBlockSize(9)         # 39
    stereo.setMinDisparity(50)       # 0
    stereo.setNumDisparities(64)    # 80
    stereo.setUniquenessRatio(55)    # 5
    stereo.setSpeckleWindowSize(41) # 41
    stereo.setSpeckleRange(1)      # 11
pass

disparity = stereo.compute(images[0].astype('uint8'),images[1].astype('uint8'))
disparity = disparity/disparity.max()*255
disparity = disparity.astype('uint8')
disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

images = np.array(images).mean(axis=0).astype('uint8')
images = cv2.applyColorMap(images, cv2.COLORMAP_JET)
    
cv2.imshow('gray', disparity)
cv2.imshow('mean', images)
cv2.imshow('false color', x.false_color)
cv2.waitKey(1)

while cv2.waitKey() != 27:
    pass
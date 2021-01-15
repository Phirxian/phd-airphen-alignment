#!/usr/bin/python3
import numpy as np
import time
import cv2

from airphen.data import *
from airphen.spectral_image import *
from glob import glob

#path = '/media/ovan/31F66DD04B004F4B1/rose/'
#folder = '20190702_110409/'
path = '/media/ovan/31F66DD04B004F4B1/database/rose4/'
folder = '20200720_121239/'
height = '1.8.npy'

filenames = glob(path+folder+'/*_450*')
filenames = [i.split('/')[-1] for i in filenames]
filenames = [i.split('_')[0] for i in filenames]

print(len(filenames))

for i in range(0, len(filenames), 3):
    start_time = time.time()
    f = filenames[i]
    
    try:
        S = SpectralImage(path, folder, f+'_', 'data/config.pkl', height)
        
        loaded, nb_kp = S.spectral_registration({
            'reference':None,
            'method':'GFTT1',
            'auto-crop-resize' : True,
            'gradient-type' : 'Ridge',
            'verbose':1,
        })
    except Exception as e:
        print(e)
        continue
    
    print('-------------------')
    print(f, 'loaded in ', time.time() - start_time)
    print('-------------------')
    
    images = np.array([i/i.max()*255 for i in loaded])
    mean = images.mean(axis=0)
    std = images.std(axis=0)*4
    
    id_blue = all_bands.index(450)
    id_red = all_bands.index(675)
    id_nir = all_bands.index(850)
    ndvi = (images[id_nir]-images[id_red]) / np.maximum(images[id_red]+images[id_nir], 0.0000001)
    ndvi = ndvi - ndvi.min()
    ndvi = ndvi / ndvi.max() * 255
    
    test = np.sqrt(std*ndvi)
    test = test-test.min()
    test = test/test.max()*255
    test = test.clip(0,255).astype('uint8')
    
    mser = cv2.MSER_create(4, _max_variation=1, _min_margin=10)
    regions = mser.detectRegions(test)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    contours = hulls
    
    mean = cv2.applyColorMap(mean.astype('uint8'), cv2.COLORMAP_JET)
    std = cv2.applyColorMap(std.clip(0,255).astype('uint8'), cv2.COLORMAP_JET)
    ndvi = cv2.applyColorMap(ndvi.clip(0,255).astype('uint8'), cv2.COLORMAP_JET)
    test = cv2.applyColorMap(test, cv2.COLORMAP_JET)
    
    false_color = S.compute_false_color()
    false_color_hsv = cv2.cvtColor(false_color, cv2.COLOR_BGR2HSV)
    
    for i,b in enumerate(loaded):
        b = false_color_normalize(b) * 255
        b = b.astype('uint8')
        cv2.imwrite('/tmp/airphen-bands-'+str(i)+'.png', b)
    
    cv2.imwrite('/tmp/airphen-std.png', std)
    cv2.imwrite('/tmp/airphen-ndvi.png', ndvi)
    cv2.imwrite('/tmp/airphen-test.png', false_color)
    #cv2.drawContours(false_color, contours, -1, (0,0,255), 1)
        
    cv2.imshow('std', std)
    cv2.imshow('ndvi', ndvi)
    cv2.imshow('false color', false_color)
    cv2.imshow('test', test)
    cv2.waitKey(1)
    
    while cv2.waitKey() != 27:
        pass
pass
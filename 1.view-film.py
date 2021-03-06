#!/usr/bin/python3
import numpy as np
import time
import cv2

from airphen.data import *
from airphen.spectral_image import *
from glob import glob

path = '/media/ovan/31F66DD04B004F4B/database/rose/'
folder = 'mais2'
height = './data/2.0.npy'

filenames = glob(path+folder+'/*_450*')
filenames = [i.split('/')[-1] for i in filenames]
filenames = [i.split('_')[0] for i in filenames]

print(len(filenames))

for i in range(0, len(filenames), 20):
    start_time = time.time()
    f = filenames[i]
    
    S = SpectralImage(path, folder, f+'_', height)
    
    loaded, nb_kp = S.spectral_registration(
        method='GFTT1',
        reference=all_bands.index(570),
        verbose=1
    )
    
    print('-------------------')
    print(f, 'loaded in ', time.time() - start_time)
    print('-------------------')
    
    images = np.array([i/i.max()*255 for i in loaded])
    mean = images.mean(axis=0)
    std = images.std(axis=0)*4
    
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
    
    cv2.drawContours(false_color, contours, -1, (0,0,255), 1)
    
    saturation = false_color_hsv[:,:,1]
    saturation = saturation-saturation.min()
    saturation = saturation/saturation.max()
    value = false_color_hsv[:,:,2]
    value = value-value.min()
    value = value/value.max()
    
    NSDVI = abs(saturation-value) / (saturation+value)
    NSDVI = NSDVI-NSDVI.min()
    NSDVI = NSDVI/NSDVI.max()*255
    NSDVI = cv2.applyColorMap(NSDVI.clip(0,255).astype('uint8'), cv2.COLORMAP_JET)
        
    cv2.imshow('NSDVI', NSDVI)
    cv2.imshow('std', std)
    cv2.imshow('ndvi', ndvi)
    cv2.imshow('test', test)
    cv2.imshow('false color', false_color)
    cv2.imshow('false color hsv', false_color_hsv)
    cv2.waitKey(1)
    
    while cv2.waitKey() != 27:
        pass
pass
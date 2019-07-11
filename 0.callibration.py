#!/usr/bin/python3
import numpy as np
import pickle as pkl
import scipy.signal as sig
import cv2

from tqdm import tqdm
from src.spectral_image import *

chessboard_shape = (13,13)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
directory = './data/steep-chess/'
bands = [450, 570, 675, 710, 730, 850]

height = [
    #'0.6/', '0.8/', '1.0/', '1.2/',
    '1.4/', '1.6/', '1.8/', '2.0/',
    '2.2/', '2.4/', '2.6/', '2.8/', '3.0/',
    '3.2/', '3.4/', '3.6/', '3.8/', '4.0/',
    '4.2/', '4.4/', '4.6/', '4.8/', '5.0/'
]

for x in range(len(height)):
    h = height[x]
    print(h)
    
    S = SpectralImage()
    imgpoints = [None] * len(bands)
    
    for s,i in enumerate(bands):
        path = directory + h + str(i) + 'nm.tif'
        image = S.read_tiff(path)
        #image = image / image.max()
        
        if i == 450:
            image = image / image.max() * 0.8
        else:
            image = image / image.max()# * 1.2
            
        image = np.clip(image * 255, 0, 255)
        image = image.astype('uint8')
        
        ret, corners = cv2.findChessboardCorners(image, chessboard_shape, None)
        if ret == True:
            corners2 = cv2.cornerSubPix(image, corners, chessboard_shape, (-1,-1), criteria)
            cv2.drawChessboardCorners(image, chessboard_shape, corners2, ret)
            imgpoints[s] = corners2
        pass
        
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        
        cv2.namedWindow('nm'+str(i), cv2.WINDOW_NORMAL)
        cv2.imshow('nm'+str(i), image)
    pass
    
    imgpoints = np.array(imgpoints)
    
    # order points !!! fix rotation error
    # because findChessboardCorners can order at 90° depending on the moon
    
    for i in range(imgpoints.shape[0]):
        imgpoints[i] = imgpoints[i, np.argsort(imgpoints[i,:,0,0]), :, :]
        imgpoints[i] = imgpoints[i, np.argsort(imgpoints[i,:,0,1]), :, :]
    pass
    
    np.save('data/' + h[0:-1] + '.npy', imgpoints)
pass

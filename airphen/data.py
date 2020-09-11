#!/usr/bin/python3
import numpy as np
import cv2

from functools import partial

height = np.array([
    #1.0, 1.2, 1.4
    1.6, 1.8, 2.0,
    2.2, 2.4, 2.6,
    2.8, 3.0, 3.2,
    3.4, 3.6, 3.8,
    4.0, 4.2, 4.4,
    4.6, 4.8, 5.0
])

# excluded green (reference)
bands_text =  ['blue',    'green',   'red',     'redge',   'redge_max', 'nir']
bands_color = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',   '#8c564b']
all_bands = [450, 570, 675, 710, 730, 850]

all_methods = [
    'ORB1',   'ORB2',   'ORB3',
    'AKAZE1', 'AKAZE2', 'AKAZE3',
    'KAZE1',  'KAZE2',  'KAZE3',
#    'BRISK1',  'BRISK2', 'BRISK3',
    'AGAST1', 'AGAST2', 'AGAST3',
    'MSER',
    'SURF1',  'SURF2',  'SURF3',
    'FAST1',  'FAST2',  'FAST3',
    'GFTT1',  'GFTT2',  'GFTT3',
]
    
detectors = {
    'ORB1'   : partial(cv2.ORB_create, nfeatures=5000),
    'ORB2'   : partial(cv2.ORB_create, nfeatures=10000),
    'ORB3'   : partial(cv2.ORB_create, nfeatures=15000),
    
    'AGAST1' : partial(cv2.AgastFeatureDetector_create, threshold=71, nonmaxSuppression=True),
    'AGAST2' : partial(cv2.AgastFeatureDetector_create, threshold=92, nonmaxSuppression=True),
    'AGAST3' : partial(cv2.AgastFeatureDetector_create, threshold=163, nonmaxSuppression=True),
    
    'AKAZE1' : partial(cv2.AKAZE_create, nOctaves=1, nOctaveLayers=1),
    'AKAZE2' : partial(cv2.AKAZE_create, nOctaves=2, nOctaveLayers=1),
    'AKAZE3' : partial(cv2.AKAZE_create, nOctaves=2, nOctaveLayers=2),
    
    'KAZE1' : partial(cv2.KAZE_create, nOctaves=4, nOctaveLayers=2),
    'KAZE2' : partial(cv2.KAZE_create, nOctaves=4, nOctaveLayers=4),
    'KAZE3' : partial(cv2.KAZE_create, nOctaves=2, nOctaveLayers=4),
    
    'MSER'  : partial(cv2.MSER_create),
    
    'BRISK1' : partial(cv2.BRISK_create, octaves=0, patternScale=.1),
    'BRISK2' : partial(cv2.BRISK_create, octaves=1, patternScale=.1),
    'BRISK3' : partial(cv2.BRISK_create, octaves=2, patternScale=.1),
    
    'SIFT'  : partial(cv2.xfeatures2d.SIFT_create, nfeatures=1000), # good ~5s
    # exact ~5s (increase parameter for higher precision)
    
    'SURF1'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=1, nOctaveLayers=1, upright=False),
    'SURF2'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=2, nOctaveLayers=1, upright=False),
    'SURF3'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=2, nOctaveLayers=2, upright=False),
    
    'FAST1'  : partial(cv2.FastFeatureDetector_create, threshold=71,  nonmaxSuppression=True),
    'FAST2'  : partial(cv2.FastFeatureDetector_create, threshold=92,  nonmaxSuppression=True),
    'FAST3'  : partial(cv2.FastFeatureDetector_create, threshold=163, nonmaxSuppression=True),
    
    'GFTT'   : partial(cv2.GFTTDetector_create, maxCorners=1000,  useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT0'  : partial(cv2.GFTTDetector_create, maxCorners=2000,  useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT1'  : partial(cv2.GFTTDetector_create, maxCorners=5000,  useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT2'  : partial(cv2.GFTTDetector_create, maxCorners=10000, useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT3'  : partial(cv2.GFTTDetector_create, maxCorners=15000, useHarrisDetector=False, minDistance=1, blockSize=9),
}

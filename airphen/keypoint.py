#!/usr/bin/python3
import numpy as np
import math
import cv2

from scipy.spatial import distance
from functools import partial

class FilterDetection:
    
    def __init__(self, matches, kp1, kp2):
        self.matches = matches
        self.kp1 = kp1
        self.kp2 = kp2
        self.a = np.array([self.kp1[i.queryIdx].pt for i in self.matches], dtype=float)
        self.b = np.array([self.kp2[i.trainIdx].pt for i in self.matches], dtype=float)
    pass
    
    def filter_distance(self, t):
        #print('filter distances ...')
        return np.array([
            i for i in range(len(self.a))
            if self.matches[i].distance < t
        ])
    pass
    
    def filter_position(self, t):
        #print('filter positions ...')
        d = np.array([
            abs(distance.euclidean(self.a[i], [0,0]) - distance.euclidean([0,0], self.b[i]))
            for i in range(len(self.a))
        ])
        
        return d<t
    pass
    
    def filter_duplicated(self, t):
        #print('filter duplicated ...')
        d = np.array([
            min([
                distance.euclidean(self.a[i], self.a[j])
                for j in range(i)
            ] + [float("inf")])
            for i in range(len(self.a))
        ])
        
        return d>t
    pass
    
    def filter_angle(self, t):
        #print('filter angles ...')
        d = np.array([
            math.atan((self.a[i][1] - self.b[i][1])/(self.b[i][0] - self.a[i][0] + 1280))*180/math.pi
            for i in range(len(self.a))
        ])
        
        return np.array([i<t for i in abs(d)])
    pass
    
pass

def keypoint_detect(img1, img2, method='FAST'):
    #print('keypoint detection ...')
    
    detectors = {
        'ORB'   : partial(cv2.ORB_create, nfeatures=5000), # error
        'AGAST' : partial(cv2.AgastFeatureDetector_create, threshold=92, nonmaxSuppression=True),
        'AKAZE' : partial(cv2.AKAZE_create),
        'KAZE'  : partial(cv2.KAZE_create),
        'MSER'  : partial(cv2.MSER_create),
        'BRISK' : partial(cv2.BRISK_create, patternScale=.1),
        #'SIFT'  : partial(cv2.xfeatures2d.SIFT_create, nfeatures=1000), # good ~5s
        # exact ~5s (increase parameter for higher precision)
        'SURF'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=2, nOctaveLayers=1, upright=False),
        'FAST'  : partial(cv2.FastFeatureDetector_create, threshold=92, nonmaxSuppression=True),
        'GFTT'  : partial(cv2.GFTTDetector_create, maxCorners=5000,useHarrisDetector=True),
    }
    
    detector = detectors[method]()
    kp1 = detector.detect(img1,None)
    kp2 = detector.detect(img2,None)
    descriptor = cv2.ORB_create()
    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)
    
    #print('keypoint matching ...')
    
    if True:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des1,des2)
    else:
        FLANN_INDEX_KDTREE = cv2.flann.FLANN_INDEX_TYPE_32F
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
        search_params = dict(checks = 2)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(des1,des2)
    pass
    
    matches = sorted(matches, key = lambda x:x.distance)
    
    return matches, kp1, kp2
pass

def keypoint_filter(matches, kp1, kp2, alpha):
    filtering = FilterDetection(matches, kp1, kp2)
    filter = np.where(filtering.filter_position(alpha))
    matches = np.array(matches)[filter]
    
    filtering = FilterDetection(matches, kp1, kp2)
    filter = np.where(filtering.filter_angle(2))
    matches = np.array(matches)[filter]
    
    #filtering = FilterDetection(matches, kp1, kp2)
    #filter = np.where(filtering.filter_duplicated(alpha*4))
    #matches = np.array(matches)[filter]
    
    return matches
pass
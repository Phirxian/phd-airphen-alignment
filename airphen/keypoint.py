#!/usr/bin/python3
import numpy as np
import itertools
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
        d = np.sum((self.a-self.b)**2, axis=1)
        
        d = np.array([
            abs(distance.euclidean(self.a[i], [0,0]) - distance.euclidean([0,0], self.b[i]))
            for i in range(len(self.a))
        ])
        
        return d<(t**2)
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
        d = 180/math.pi*np.arctan(
            (self.a[:,1] - self.b[:,1])/
            (self.b[:,0] - self.a[:,0] + 1280)
        )
        
        return abs(d)<t
    pass
    
pass
    
detectors = {
    'ORB1'   : partial(cv2.ORB_create, nfeatures=5000),
    'ORB2'   : partial(cv2.ORB_create, nfeatures=10000),
    'ORB3'   : partial(cv2.ORB_create, nfeatures=15000),
    #
    #'AGAST1' : partial(cv2.AgastFeatureDetector_create, threshold=71, nonmaxSuppression=True),
    #'AGAST2' : partial(cv2.AgastFeatureDetector_create, threshold=92, nonmaxSuppression=True),
    #'AGAST3' : partial(cv2.AgastFeatureDetector_create, threshold=163, nonmaxSuppression=True),
    #
    #'AKAZE1' : partial(cv2.AKAZE_create, nOctaves=1, nOctaveLayers=1),
    #'AKAZE2' : partial(cv2.AKAZE_create, nOctaves=2, nOctaveLayers=1),
    #'AKAZE3' : partial(cv2.AKAZE_create, nOctaves=2, nOctaveLayers=2),
    #
    #'KAZE1' : partial(cv2.KAZE_create, nOctaves=4, nOctaveLayers=2),
    #'KAZE2' : partial(cv2.KAZE_create, nOctaves=4, nOctaveLayers=4),
    #'KAZE3' : partial(cv2.KAZE_create, nOctaves=2, nOctaveLayers=4),
    #
    #'MSER'  : partial(cv2.MSER_create),
    #
    #'BRISK1' : partial(cv2.BRISK_create, octaves=0, patternScale=.1),
    #'BRISK2' : partial(cv2.BRISK_create, octaves=1, patternScale=.1),
    #'BRISK3' : partial(cv2.BRISK_create, octaves=2, patternScale=.1),
    
    #'SIFT'  : partial(cv2.xfeatures2d.SIFT_create, nfeatures=1000), # good ~5s
    # exact ~5s (increase parameter for higher precision)
    
    #'SURF1'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=1, nOctaveLayers=1, upright=False),
    #'SURF2'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=2, nOctaveLayers=1, upright=False),
    #'SURF3'  : partial(cv2.xfeatures2d.SURF_create, hessianThreshold=10, nOctaves=2, nOctaveLayers=2, upright=False),
    
    'FAST1'  : partial(cv2.FastFeatureDetector_create, threshold=71,  nonmaxSuppression=True),
    'FAST2'  : partial(cv2.FastFeatureDetector_create, threshold=92,  nonmaxSuppression=True),
    'FAST3'  : partial(cv2.FastFeatureDetector_create, threshold=163, nonmaxSuppression=True),
    
    'GFTT'   : partial(cv2.GFTTDetector_create, maxCorners=1000,  useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT0'  : partial(cv2.GFTTDetector_create, maxCorners=2000,  useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT1'  : partial(cv2.GFTTDetector_create, maxCorners=5000,  useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT2'  : partial(cv2.GFTTDetector_create, maxCorners=10000, useHarrisDetector=False, minDistance=1, blockSize=9),
    'GFTT3'  : partial(cv2.GFTTDetector_create, maxCorners=15000, useHarrisDetector=False, minDistance=1, blockSize=9),
}

def keypoint_detect(img1, img2, method='GFTT'):
    #print('keypoint detection ...')
    
    detector = detectors[method]()
    kp1, kp2 = detector.detect([img1, img2],None)
    
    descriptor = cv2.ORB_create()
    (kp1, kp2), (des1, des2) = descriptor.compute([img1, img2], [kp1, kp2])
    
    #print('keypoint matching ...')
    matcher = 0
    
    if matcher==0:
        bf = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=True)
        matches = bf.match(des1,des2)
    elif matcher==1:
        fv1 = cv2.detail.computeImageFeatures2(descriptor,img1)
        fv2 = cv2.detail.computeImageFeatures2(descriptor,img2)
        bf = cv2.detail.BestOf2NearestMatcher_create(True, 0.01,0,0)
        matches = bf.apply(fv1, fv2).getMatches()
        print(len(matches))
    else:
        FLANN_INDEX_KDTREE = cv2.flann.FLANN_INDEX_TYPE_32F
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.match(np.asarray(des1, np.float32), np.asarray(des2, np.float32))
    pass
    
    #matches = sorted(matches, key = lambda x:x.distance)
    
    return matches, kp1, kp2
pass

def keypoint_filter(matches, kp1, kp2, alpha):
    filtering = FilterDetection(matches, kp1, kp2)
    filter = np.where(filtering.filter_position(alpha))
    matches = np.array(matches)[filter]
    
    filtering = FilterDetection(matches, kp1, kp2)
    filter = np.where(filtering.filter_angle(1))
    matches = np.array(matches)[filter]
    
    #filtering = FilterDetection(matches, kp1, kp2)
    #filter = np.where(filtering.filter_duplicated(alpha*4))
    #matches = np.array(matches)[filter]
    
    return matches
pass
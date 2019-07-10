#!/usr/bin/python3
import numpy as np
import math
import cv2

from scipy.spatial import distance

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
            #distance.cosine(self.a[i], self.b[i]) * 180 / np.pi
            #distance.cityblock(self.a[i], self.b[i])
            #distance.euclidean(self.a[i], self.b[i])
            abs(distance.euclidean(self.a[i], [0,0]) - distance.euclidean([0,0], self.b[i]))
            for i in range(len(self.a))
        ])
        
        return d<t
    pass
    
    def filter_duplicated(self, t):
        #print('filter duplicated ...')
        d = np.array([
            #distance.cosine(self.a[i], self.b[i]) * 180 / np.pi
            #distance.cityblock(self.a[i], self.b[i])
            #distance.euclidean(self.a[i], self.b[i])
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
            #math.atan2(self.a[i][1] - self.b[i][1], self.a[i][0] - self.b[i][0] + 1280) * 180 / np.pi
            math.atan((self.a[i][1] - self.b[i][1])/(self.b[i][0] - self.a[i][0] + 1280))*180/math.pi
            for i in range(len(self.a))
        ])
        
        d = abs(d)
        
        return np.array([i<t for i in d])
    pass
    
pass

def keypoint_detect(img1, img2):
    #print('keypoint detection ...')
    #detector = cv2.ORB_create(nfeatures=1000) # error
    #detector = cv2.AKAZE_create() # good but slow ~15s
    #detector = cv2.BRISK_create(patternScale=0.1)
    #detector = cv2.xfeatures2d.SIFT_create(nfeatures=1000) # good ~5s
    
    # exact ~5s (increase parameter for higher precision)
    detector = cv2.xfeatures2d.SURF_create(nOctaves=1, nOctaveLayers=1, upright=False)
    kp1 = detector.detect(img1,None)
    kp2 = detector.detect(img2,None)
    
    #descriptor = detector
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
    filter = np.where(filtering.filter_angle(1))
    matches = np.array(matches)[filter]
    
    #filtering = FilterDetection(matches, kp1, kp2)
    #filter = np.where(filtering.filter_duplicated(alpha*4))
    #matches = np.array(matches)[filter]
    
    return matches
pass

def keypoint_draw(img1, img2, kp1, kp2, matches):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for i in kp1:
        px = np.int0(i.pt)
        img1[px[1], px[0], :] = [0,0,255]
    
    for i in kp2:
        px = np.int0(i.pt)
        img2[px[1], px[0], :] = [0,0,255]
    
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
pass
#!/usr/bin/python3
from collections import namedtuple
import numpy as np
import math
import cv2

from tqdm import tqdm
from .keypoint import *
from .spanning_tree import *
from .image_processing import *
from .debug import *

Detection = namedtuple('Detection', ('kp', 'des'))

# @ref the best reference : default=1=570 !! (maximum number of matches)
def multiple_refine_allignement(prefix, loaded, method='SURF', sink=0, verbose=True):
    ######################### image transformation
    
    img = [ i.astype('float32') for i in loaded]
    img = [ gradient_normalize(i, 0.01) for i in img]
    grad = [ build_gradient(i, method='Ridge') for i in img]
    
    ######################### keypoint detection and description
    
    descriptor = cv2.ORB_create()
    detector = detectors[method]()
    
    if verbose:
        print('keypoint detection')
        
    keypoints = []
    for i in range(0, len(loaded)):
        kp = detector.detect(grad[i],None)
        kp, des = descriptor.compute(grad[i], kp)
        keypoints.append(Detection(kp, des))
        cv2.imshow('grad'+str(i), grad[i])
        
    ######################### keypoint matching for each pairs
    
    bf = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=True)
    evaluators = [cv2.RANSAC, cv2.LMEDS, cv2.RHO]
    max_dist = 20
    
    matches = [[]] * len(loaded)
    transforms = [[]] * len(loaded)
    arcs = []
    
    if verbose:
        print('keypoint graph matching')
        
    for i in tqdm(range(0, len(loaded))):
        for j in range(0, len(loaded)):
            if i == j and i>j: continue
            # kp1 = 'ref'==to kp2='reg'
            kp1, kp2 = keypoints[i].kp, keypoints[j].kp
            M = bf.match(keypoints[i].des, keypoints[j].des)
            M = keypoint_filter(M, kp1, kp2, max_dist)
            
            source = np.array([kp1[k.queryIdx].pt for k in M], dtype=float)
            target = np.array([kp2[k.trainIdx].pt for k in M], dtype=float)
            mat = [cv2.DMatch(k,k,0) for k in range(len(source))]
            source = source[np.newaxis, :].astype('int')
            target = target[np.newaxis, :].astype('int')
            T, mask = cv2.findHomography(target, source, evaluators[2])
            #target = np.float32(target[0,mask,:])
            #source = np.float32(source[0,mask,:])
            
            if T is not None:
                arcs.append(Arc(tail=j, head=i, weight=len(M)))
                transforms[i].append(T)
                matches[i].append(M)
            
    ######################### getting the best registration scheme
    
    if verbose:
        print('registration scheme')
    max_tree = gen_spanning_arborescence(arcs, sink, 'max')
    max_tree = [i for i in max_tree.values()]
    print(max_tree)
    
    ######################### standar alignement
    
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    center = np.array(dsize)/2
    bbox, centers, keypoint_found = [None]*len(loaded), [None]*len(loaded), [None]*len(loaded)
    
    bbox[sink] = [0,0,dsize[1],dsize[0]]
    centers[sink] = [0,0]
    
    if verbose:
        print('registration !')
        
    for link in max_tree:
        print('registration ', link.tail, sink)
        c, M = link.tail, np.eye(3)
        while c != sink:
            l = [i for i in max_tree if i.tail==c][0]
            M = np.matmul(M,transforms[l.head][l.tail])
            c = l.head
        print(M)
            
        loaded[link.tail] = cv2.warpPerspective(loaded[link.tail], M, dsize)
        bbox[link.tail] = get_perspective_min_bbox(M, loaded[sink])
        centers[link.tail] = cv2.perspectiveTransform(np.array([[center]]),M)[0,0] - center
        pass
    pass
    
    return loaded, np.array(bbox), keypoint_found, np.array(centers)
pass
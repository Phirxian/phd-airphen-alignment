#!/usr/bin/python3
from collections import namedtuple
import numpy as np
import math
import cv2

from tqdm import tqdm
from .keypoint import *
from .spanning_tree import *
from .image_processing import *
from .data import *

class Detection:
    def __init__(self,kp, des, T):
        self.kp = kp
        self.des = des
        self.T = T
        self.nb_kp = 0
        

def global_kp_extractor(grad, detector, descriptor, verbose):
    if verbose: print('keypoint detection')
    keypoints = []
    for i in range(0, len(grad)):
        kp = detector.detect(grad[i],None)
        kp, des = descriptor.compute(grad[i], kp)
        keypoints.append(Detection(kp, des, None))
        #cv2.imshow('grad'+str(i), grad[i])
    return keypoints
    
def kp_graph_matching(keypoints, descriptor, iterator, verbose, max_dist=20):
    bf = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=True)
    evaluators = [cv2.RANSAC, cv2.LMEDS, cv2.RHO]
    arcs = []
    
    if verbose: print('keypoint graph matching')
        
    for j,i in tqdm(iterator):
        # kp1 = 'ref'==to kp2='reg'
        kp1, kp2 = keypoints[i].kp, keypoints[j].kp
        M = bf.match(keypoints[i].des, keypoints[j].des)
        M = keypoint_filter(M, kp1, kp2, max_dist)
        
        source = np.array([kp1[k.queryIdx].pt for k in M], dtype=float)
        target = np.array([kp2[k.trainIdx].pt for k in M], dtype=float)
        source = source[np.newaxis, :].astype('int')
        target = target[np.newaxis, :].astype('int')
        T, mask = cv2.findHomography(target, source, evaluators[0])
        target = np.float32(target[0,mask,:])
        source = np.float32(source[0,mask,:])
        
        if T is not None:
            corrected = cv2.perspectiveTransform(target,T)
            diff = corrected-source
            l2 = np.sqrt((diff**2).sum(axis=1))
            keypoints[j].T = T
            keypoints[j].nb_kp = len(source)
            A = Arc(tail=j, head=i, weight=l2.mean())
            arcs.append(A)
                
    return arcs
    
def apply_spanning_registration(arcs, keypoints, sink, loaded, verbose):
    tree = gen_spanning_arborescence(arcs, sink, 'min')
    tree = [i for i in tree.values()]
    #print(tree)
    
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    center = np.array(dsize)/2
    bbox, centers, keypoint_found = [None]*len(loaded), [None]*len(loaded), [None]*len(loaded)
    
    bbox[sink] = [0,0,dsize[1],dsize[0]]
    centers[sink] = [0,0]
        
    for link in tree:
        if verbose:
            print('registration ', link.tail, link.head, link.weight, 'to sink', sink)
        
        c, M = link.tail, np.eye(3)
        while c != sink:
            l = [i for i in tree if i.tail==c][0]
            M = np.matmul(keypoints[c].T,M)
            c = l.head
            
        loaded[link.tail] = cv2.warpPerspective(loaded[link.tail], M, dsize)
        bbox[link.tail] = get_perspective_min_bbox(M, loaded[sink])
        centers[link.tail] = cv2.perspectiveTransform(np.array([[center]]),M)[0,0] - center
        keypoint_found[link.tail] = keypoints[link.tail].nb_kp
        
    return loaded, np.array(bbox), keypoint_found, np.array(centers)

# @ref the best reference : default=1=570 !! (maximum number of matches)
def multiple_refine_allignement(loaded, method, iterator, sink, verbose=True):
    ######################### image transformation
    
    img = [i.astype('float32') for i in loaded]
    img = [gradient_normalize(i, 0.01) for i in img]
    grad = [build_gradient(i, method='Ridge') for i in img]
    
    ######################### keypoint detection and description
    
    descriptor = cv2.ORB_create()
    detector = detectors[method]()
    
    keypoints = global_kp_extractor(grad, detector, descriptor, verbose)
    arcs = kp_graph_matching(keypoints, descriptor, iterator, verbose)
    
    return apply_spanning_registration(arcs, keypoints, sink, loaded, verbose)

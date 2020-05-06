#!/usr/bin/python3
import numpy as np
import math
import cv2

from .keypoint import *
from .image_processing import *
from .debug import *

# @ref the best reference : default=1=570 !! (maximum number of matches)
def refine_allignement(prefix, loaded, method='SURF', ref=1, verbose=True):
    img = [ i.astype('float32') for i in loaded]
    img = [ gradient_normalize(i, 0.001) for i in img]
    grad = [ build_gradient(i, method='Ridge').astype('uint8') for i in img]
    img = [ (i/i.max()*255).clip(0,255).astype('uint8') for i in img]
    
    # identify transformation for each band to the next one
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    bbox, centers, keypoint_found = [], [], []
    max_dist = 10
    
    for i in range(0, len(loaded)):
        if i == ref:
            keypoint_found.append(-1)
            centers.append(np.array([0,0]))
            continue
            
        if verbose > 2:
            print('spectral registration', i)
    
        matches, kp1, kp2 = keypoint_detect(grad[ref], grad[i], method)
        matches = keypoint_filter(matches, kp1, kp2, max_dist)
        
        if len(matches) < 2:
            keypoint_found.append(-1)
            centers.append(np.array([0,0]))
            continue
            
        #print('estimate transformation ...')
        
        source = np.array([kp1[i.queryIdx].pt for i in matches], dtype=float)
        target = np.array([kp2[i.trainIdx].pt for i in matches], dtype=float)
        matches = [cv2.DMatch(i,i,0) for i in range(len(source))]
        
        source = source[np.newaxis, :].astype('int')
        target = target[np.newaxis, :].astype('int')
        
        #tr = cv2.createThinPlateSplineShapeTransformer()
        ##tr = cv2.createAffineTransformer(fullAffine=False)
        #tr.estimateTransformation(source,target,matches)
        #loaded[i] = tr.warpImage(loaded[i])
        
        evaluators = [cv2.RANSAC, cv2.LMEDS, cv2.RHO]
        M, mask = cv2.findHomography(target, source, evaluators[0])
        
        if M is None:
            keypoint_found.append(-1)
            centers.append(np.array([0,0]))
            continue
        
        loaded[i] = cv2.warpPerspective(loaded[i], M, dsize)
        bbox.append(get_perspective_min_bbox(M, loaded[ref]))
        #centers.append(cv2.perspectiveTransform(np.array([[[0.,0.]]]),M)[0,0])
        
        center = np.array(dsize)/2
        centers.append(cv2.perspectiveTransform(np.array([[center]]),M)[0,0] - center)
        
        mask = np.where(mask.flatten())
        target = np.float32(target[0,mask,:])
        source = np.float32(source[0,mask,:])
        keypoint_found.append(len(mask[0]))
        #keypoint_found.append(source.shape[1])
        
        if verbose > 0:
            corrected = cv2.perspectiveTransform(target,M)
            diff = corrected-source
            
            l2 = np.sqrt((diff**2)[0].sum(axis=1))
            #rmse = np.sqrt((diff**2).mean())
            mean = l2.mean()
            min = l2.min()
            max = l2.max()
            std = l2.std()
            
            if verbose > 1:
                scatter_plot_residual(prefix, source, target, corrected, loaded[0].shape, i)
                
            if verbose > 2:
                draw_final_keypoint_matching(source, target, grad[ref], grad[i])
            
            print(
                f'points={str(target.shape[1]).ljust(4)} ; '
                f'l2={str(np.round(mean, 5)).ljust(7)} ; '
                f'std={str(np.round(std, 5)).ljust(7)} ; '
                f'min={str(np.round(min, 5)).ljust(7)} ; '
                f'max={str(np.round(max, 5)).ljust(7)}'
            )
        pass
    pass
    
    #print(centers)
        
    return loaded, np.array(bbox), keypoint_found, np.array(centers)
pass
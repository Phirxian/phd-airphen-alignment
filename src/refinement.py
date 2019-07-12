#!/usr/bin/python3
import numpy as np
import math
import cv2

from .keypoint import *
from .debug import *

def normalize(i):
    G = cv2.GaussianBlur(i,(9,9),cv2.BORDER_DEFAULT)
    return i/(G+1)*255
pass

def build_gradient(img, scale = 0.15, delta=0, ddepth = cv2.CV_32F):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    grad_x = cv2.Scharr(img, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Scharr(img, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad = grad.astype('float')**2
    grad = grad/grad.max()*255
    grad = grad.reshape([*grad.shape, 1])
    grad = grad.astype('uint8')
    grad = clahe.apply(grad)
    return grad.astype('uint8')
pass

def get_perspective_min_bbox(M, img, p=2):
    h,w = img.shape[:2]
    
    pts_x = np.float32([[  p,  p], [  p, h-p]]).reshape(-1,1,2)
    pts_y = np.float32([[w-p,h-p], [w-p,   p]]).reshape(-1,1,2)
    
    coords_x = cv2.perspectiveTransform(pts_x,M)[:,0,:]
    coords_y = cv2.perspectiveTransform(pts_y,M)[:,0,:]
    
    [xmin, xmax] = max(coords_x[0,0], coords_x[1,0]), min(coords_y[0,0], coords_y[1,0])
    [ymin, ymax] = max(coords_x[0,1], coords_y[1,1]), min(coords_x[1,1], coords_y[0,1])
    
    return np.int32([ymin,xmin,ymax,xmax])
pass

def refine_allignement(loaded, verbose=True):
    img = [ i.astype('float32') for i in loaded]
    img = [ normalize(i) for i in img]
    grad = [ build_gradient(i).astype('uint8') for i in img]
    img = [ (i/i.max()*255).astype('uint8') for i in img]
    
    identity = np.array([[1,0,0], [0,1,0]], dtype='float32')
    transform = [identity] * len(loaded)
    
    # identify transformation for each band to the next one
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    mid = 1 # the best reference !! (maximum number of matches)
    max_dist = 10
    bbox = []
    
    for i in range(0, len(loaded)):
        if i == mid:
            continue
    
        matches, kp1, kp2 = keypoint_detect(grad[mid], grad[i])
        matches = keypoint_filter(matches, kp1, kp2, max_dist)
        
        if len(matches) < 2:
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
        
        M, mask = cv2.findHomography(target, source, cv2.RANSAC)
        loaded[i] = cv2.warpPerspective(loaded[i], M, dsize)
        bbox.append(get_perspective_min_bbox(M, loaded[mid]))
        
        if verbose:
            mask = np.where(mask.flatten())
            target = np.float32(target[0,mask,:])
            source = np.float32(source[0,mask,:])
            
            corrected = target
            corrected = cv2.perspectiveTransform(corrected,M)
            diff = corrected-source
            
            l2 = np.sqrt((diff**2)[0].sum(axis=1))
            #rmse = np.sqrt((diff**2).mean())
            mean = l2.mean()
            min = l2.min()
            max = l2.max()
            std = l2.std()
            
            if False and i == 3:
                scatter_plot_residual(source, target, corrected)
                #draw_final_keypoint_matching(source, target, grad[mid], grad[i])
            
            print(
                f'points={str(target.shape[1]).ljust(4)} ; '
                f'l2={str(np.round(mean, 5)).ljust(7)} ; '
                f'std={str(np.round(std, 5)).ljust(7)} ; '
                f'min={str(np.round(min, 5)).ljust(7)} ; '
                f'max={str(np.round(max, 5)).ljust(7)}'
            )
        pass
    pass
        
    return loaded, np.array(bbox)
pass
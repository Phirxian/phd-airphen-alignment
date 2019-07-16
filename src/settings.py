import cv2
import numpy as np

from glob import glob
from natsort import natsorted
from .refinement import *
from .data import *

def normalize(i):
    G = cv2.GaussianBlur(i,(9,9),cv2.BORDER_DEFAULT)
    i = abs(i+G.min()) / G.max()
    return i
pass

def compute_false_color(S):
    img = np.zeros((S[0].shape[0], S[0].shape[1], 3))
    img[:,:,0] = normalize(S[0])*92  # B
    img[:,:,1] = normalize(S[1])*200 # G
    img[:,:,2] = normalize(S[2])*200 # R
    return img.astype('uint8')
pass

def crop_all(S, loaded, min_xy, max_xy):
    S.ground_thrust = S.ground_thrust[min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]
    for i in range(len(loaded)):
        loaded[i] = loaded[i][min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]
    pass
pass

def affine_transform(S, loaded):
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    centroid = S.chessboard[:,:,0,:].mean(axis=0).astype('float32') 
    transform = [None] * len(loaded)
    
    for i in range(len(loaded)):
        points = S.chessboard[i,:,0,:]
        if cv2.__version__[0] == '4': transform[i] = cv2.estimateAffine2D(points, centroid)[0]
        else: transform[i] = cv2.estimateRigidTransform(points, centroid, fullAffine=False)
        loaded[i] = cv2.warpAffine(loaded[i], transform[i], dsize)
    pass
    
    return loaded, np.array(transform)
pass

def load_spectral_bands(S, verbose=0, method='SURF', reference=1):
    loaded = S.loaded

    # alligne trough affine transfrom using pre-callibration
    loaded, transform = affine_transform(S, loaded)
    max_xy = np.min(transform[:,:,2], axis=0).astype('int')
    min_xy = np.max(transform[:,:,2], axis=0).astype('int')
    crop_all(S, loaded, np.flip(min_xy), np.flip(max_xy))
    
    # refine allignment with homography
    loaded, bbox, nb_kp = refine_allignement(loaded, method, reference, verbose)
    min_xy = np.max(bbox[:, :2], axis=0).astype('int')
    max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
    crop_all(S, loaded, min_xy, max_xy)
    
    return loaded, nb_kp
pass

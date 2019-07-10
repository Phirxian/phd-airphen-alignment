import cv2
import numpy as np

from glob import glob
from natsort import natsorted
from refinement import *

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

def affine_transform(S, current, loaded):
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    
    # order points !!! fix rotation error
    # because findChessboardCorners can order at 90° depending on the moon
    
    for i in range(current.shape[0]):
        current[i] = current[i, np.argsort(current[i,:,0,0]), :, :]
        current[i] = current[i, np.argsort(current[i,:,0,1]), :, :]
    pass
    
    centroid = current[:,:,0,:].mean(axis=0).astype('float32') 
    transform = [None] * len(loaded)
    
    for i in range(len(loaded)):
        points = current[i,:,0,:]
        if cv2.__version__[0] == '4': transform[i] = cv2.estimateAffine2D(points, centroid)[0]
        else: transform[i] = cv2.estimateRigidTransform(points, centroid, fullAffine=False)
        loaded[i] = cv2.warpAffine(loaded[i], transform[i], dsize)
    pass
    
    return loaded, np.array(transform)
pass

def load_spectral_bands(S, set, subset, prefix='', height=None):
    path = set + '/' + str(subset) + '/'
    
    bands = [450, 570, 675, 710, 730, 850]
    loaded = [S.read_tiff(path + prefix + str(i) + 'nm.tif') for i in bands]
    
    S.ground_thrust = cv2.imread(path + 'mask.jpg')
    if S.ground_thrust is None:
        S.ground_thrust = np.zeros((loaded[0].shape[0], loaded[0].shape[1], 3))
    
    # nearest chessboard points
    current = np.load('./data/'+(subset if height == None else height)+'.npy').astype('float32')
    
    # alligne trough affine transfrom using pre-callibration
    loaded, transform = affine_transform(S, current, loaded)
    
    # crop first allignement
    max_xy = np.min(transform[:,:,2], axis=0).astype('int')
    min_xy = np.max(transform[:,:,2], axis=0).astype('int')
    crop_all(S, loaded, np.flip(min_xy), np.flip(max_xy))
    
    # refine allignment with homography
    loaded, bbox = refine_allignement(loaded)
    
    # crop refinement
    min_xy = np.max(bbox[:, :2], axis=0).astype('int')
    max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
    crop_all(S, loaded, min_xy, max_xy)
    
    # set
    
    S.images = {
        'redge_max' : loaded[4], 'redge' : loaded[3], 
        'nir'       : loaded[5], 'red'   : loaded[2],
        'green'     : loaded[1], 'blue'  : loaded[0],
    }

    S.false_color = compute_false_color(loaded)
pass

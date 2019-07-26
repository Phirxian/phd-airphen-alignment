import cv2
import numpy as np
import rasterio
import math
import os

def gradient_normalize(i):
    s = math.ceil(i.shape[0]**0.4) // 2 * 2 +1
    G = cv2.GaussianBlur(i,(s,s),cv2.BORDER_DEFAULT)
    return i/(G+1)*255
pass

def false_color_normalize(i):
    s = math.ceil(i.shape[0]**0.4) // 2 * 2 +1
    G = cv2.GaussianBlur(i,(s,s),cv2.BORDER_DEFAULT)
    i = abs(i+G.min()) / G.max()
    return i.clip(0,1)
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
    
def read_tiff(fname):
    if not os.path.exists(fname): return None
    # skep detection if the corresponding output exist
    #if os.path.exists(csv): continue
    
    geotiff = rasterio.open(fname)
    data = geotiff.read()
    tr = geotiff.transform
    
    return data[0]
pass
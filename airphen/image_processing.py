import cv2
import numpy as np
import rasterio
import math
import os

from numpy.fft import fft2, ifft2, fftshift
import scipy.ndimage.interpolation as ndii

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

def build_gradient(img, scale = 0.15, delta=0, method='Scharr'):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    
    if method == 'Sobel':
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif method == 'Laplacian':
        grad = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    elif method == 'Canny':
        grad = cv2.Canny(img.astype('uint8'), delta, 255)
    else:
        grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    pass
    
    grad = cv2.convertScaleAbs(grad)
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

def translation(im0, im1):
    """Return translation vector to register images."""
    shape = im0.shape
    f0 = fft2(im0)
    f1 = fft2(im1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), shape)
    if t0 > shape[0] // 2: t0 -= shape[0]
    if t1 > shape[1] // 2: t1 -= shape[1]
    return [t0, t1]
pass

def perspective_similarity_transform(S, loaded, ref):
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    
    img = [ i.astype('float32') for i in loaded]
    img = [ gradient_normalize(i) for i in img]
    grad = [ build_gradient(i).astype('uint8') for i in img]
    bbox = []
    
    src = np.array([
        [       0,        0],
        [       0, dsize[1]],
        [dsize[0],        0],
        [dsize[0], dsize[1]],
    ], np.float32)
    
    s=50
    d=250
    
    for i in range(len(loaded)):
        if i == ref:
            continue
        
        upperleft = translation(grad[i][s:d, s:d], grad[ref][s:d, s:d])
        upperright = translation(grad[i][s:d, -d:-s], grad[ref][s:d, -d:-s])
        lowerleft = translation(grad[i][-d:-s, s:d], grad[ref][-d:-s, s:d])
        lowerright = translation(grad[i][-d:-s, -d:-s], grad[ref][-d:-s, -d:-s])
        
        dst = src.copy()
        dst[0] -= upperleft
        dst[1] -= upperright
        dst[2] -= lowerleft
        dst[3] -= lowerright
        
        M = cv2.getPerspectiveTransform(src, dst)
        bbox.append(get_perspective_min_bbox(M, loaded[ref]))
        loaded[i] = cv2.warpPerspective(loaded[i], M, dsize)
    pass
    
    return loaded, np.array(bbox)
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
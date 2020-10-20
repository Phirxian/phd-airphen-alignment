import cv2
import numpy as np
import rasterio
import math
import os

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy.fft import fft2, ifft2, fftshift
import scipy.ndimage.interpolation as ndii

def gradient_normalize(i, q=0.01):
    if q is None:
        s = math.ceil(i.shape[0]**0.4) // 2 * 2 +1
        G = cv2.GaussianBlur(i,(s,s),cv2.BORDER_DEFAULT)
        if G is None:
            raise ValueError('image or blur is None')
        return i/(G+1)*255
    else:
        min = np.quantile(i, q)
        max = np.quantile(i, 1-q)
        i = (i-min) / (max-min) * 255
        return i.clip(0,255)
        

def false_color_normalize(i, q=0.01):
    min = np.quantile(i, q)
    max = np.quantile(i, 1-q)
    i = (i-min) / (max-min)
    return i.clip(0,1)
    

def build_gradient(img, scale = 0.15, delta=0, method='Scharr'):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    if method == 'Sobel':
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad = gradient_normalize(grad, q=0.001)
    elif method == 'Laplacian':
        grad = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    elif method == 'Canny':
        grad = cv2.Canny(img.astype('uint8'), 125, 125)
    elif method == 'Ridge':
        H_elems = hessian_matrix(img, sigma=1-scale, order='rc')
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
        grad = minima_ridges
        grad = 255-gradient_normalize(grad, q=0.001)
    else:
        grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    grad = cv2.convertScaleAbs(grad)
    grad = grad.astype('uint8')
    grad = clahe.apply(grad)
    grad = grad.astype('float')**2
    grad = grad/grad.max()*255
    grad = grad.reshape([*grad.shape, 1])
    
    return grad.astype('uint8')
    

def get_perspective_min_bbox(M, img, p=2):
    h,w = img.shape[:2]
    
    pts_x = np.float32([[  p,  p], [  p, h-p]]).reshape(-1,1,2)
    pts_y = np.float32([[w-p,h-p], [w-p,   p]]).reshape(-1,1,2)
    
    coords_x = cv2.perspectiveTransform(pts_x,M)[:,0,:]
    coords_y = cv2.perspectiveTransform(pts_y,M)[:,0,:]
    
    [xmin, xmax] = max(coords_x[0,0], coords_x[1,0]), min(coords_y[0,0], coords_y[1,0])
    [ymin, ymax] = max(coords_x[0,1], coords_y[1,1]), min(coords_x[1,1], coords_y[0,1])
    
    return np.int32([ymin,xmin,ymax,xmax])
  

def crop_all(S, loaded, min_xy, max_xy, crop_ground_thrust=False):
    if crop_ground_thrust:
        S.ground_thrust = S.ground_thrust[min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]
    for i in range(len(loaded)):
        loaded[i] = loaded[i][min_xy[0]:max_xy[0], min_xy[1]:max_xy[1]]
    

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
    

def affine_transform_linear(S, loaded):
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    transform = [None] * len(loaded)
    
    def eval_model(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
        
    # mean for factor $a,b,c,d$
    rotation_scale = [
        #[ 1.00338690e+00, -2.36790400e-04,  7.42199784e-04,  1.00322241e+00],
        #[ 9.96668519e-01, -6.35845767e-04,  1.33195202e-03,  9.97611207e-01],
        #[ 1.00287955e+00, -1.44683277e-04, -2.50432200e-04,  1.00280779e+00],
        #[ 9.98396830e-01,  2.36257574e-03, -1.61286382e-03,  9.97841848e-01],
        #[ 9.97775109e-01, -4.42514896e-04,  2.19043135e-04,  9.96903476e-01],
        #[ 1.00091705e+00, -9.09167955e-04, -4.29086670e-04,  1.00163444e+00],
        [ 1.00166648e+00, -5.07480271e-04,  1.86119466e-03,  1.01072014e+00],
        [ 9.98065617e-01, -3.02441015e-04,  1.39604708e-03,  9.84364903e-01],
        [ 1.00605485e+00,  1.43150803e-04, -4.62634090e-04,  1.00371692e+00],
        [ 9.95644883e-01,  1.76160468e-03, -2.18545206e-03,  1.00609635e+00],
        [ 9.96589887e-01, -1.16663007e-03,  8.57040497e-05,  9.90016350e-01],
        [ 1.00197647e+00,  6.00443125e-05, -7.45070370e-04,  1.00547568e+00],
    ]
    
    translation_model_params = [
        #([-2.14341005e-01,  3.06476550e+00, -1.56835580e+01,  2.65363529e+01], [ 4.89008103e-02, -5.43007565e-01,  1.81760798e+00, -5.09998030e+01]),
        #([-2.97328285e-01,  3.78134481e+00, -1.74477307e+01,  3.57275192e+01], [-4.00242147e-01,  5.23050713e+00, -2.47586203e+01,  7.91007266e+01]),
        #([ 2.85766303e-01, -3.75413817e+00,  1.76793803e+01, -2.42718058e+01], [ 1.02607225e-02, -5.42365651e-02, -4.57016171e-02,  3.07288625e+01]),
        #([-3.03964423e-01,  3.92977267e+00, -1.82182608e+01,  3.76066731e+01], [ 3.81823772e-01, -5.05192154e+00,  2.41896920e+01, -1.06395076e+02]),
        #([ 2.82467733e-01, -3.64513371e+00,  1.71317668e+01, -4.89568592e+01], [ 3.46691296e-01, -4.67803690e+00,  2.30067763e+01, -2.33119194e+01]),
        #([ 2.73308578e-01, -3.63229467e+00,  1.73784966e+01, -2.77109446e+01], [-4.10248022e-01,  5.34334908e+00, -2.51084115e+01,  7.18093591e+01]),
        ([ 1.45231374e-01, -1.53865271e-01, -6.04457629e+00,  2.93003050e+01], [1.877406750e-01, -1.79194905e+00,  5.73067138e+00, -6.39449547e+01]),
        ([-3.37892829e-01,  3.97680127e+00, -1.72713952e+01,  2.82637366e+01], [-4.50082205e-01,  5.57780655e+00, -2.51331559e+01,  8.49939648e+01]),
        ([-9.33922708e-02, -4.04531546e-01,  7.88138007e+00, -2.78105057e+01], [-7.24194437e-02,  6.76124886e-01, -2.35982747e+00,  3.46986328e+01]),
        ([ 1.58868418e-01, -2.87997447e-01, -5.38329042e+00,  3.96119093e+01], [ 4.95126108e-01, -6.02705211e+00,  2.68857988e+01, -1.12907265e+02]),
        ([ 4.35827375e-01, -4.79335788e+00,  1.92749887e+01, -4.39302736e+01], [ 4.16267497e-01, -5.19876321e+00,  2.39572982e+01, -1.51467125e+01]),
        ([-1.89352586e-01,  5.28746256e-01,  4.97412807e+00, -2.89374049e+01], [-5.41234152e-01,  6.47288460e+00, -2.83785174e+01,  7.00658948e+01]),
    ]
    
    for i in range(len(loaded)):
        x = eval_model(S.height, *translation_model_params[i][0])
        y = eval_model(S.height, *translation_model_params[i][1])
        transform[i] = np.array([
            [rotation_scale[i][0], rotation_scale[i][1], x],
            [rotation_scale[i][2], rotation_scale[i][3], y],
        ])
        loaded[i] = cv2.warpAffine(loaded[i], transform[i], dsize)
    pass
    
    print(np.array(transform))
    
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
    
    s, d = 10, 200
    
    for i in range(len(loaded)):
        if i == ref:
            continue
        
        dst = src.copy()
        dst[0] -= translation(grad[i][s:d, s:d], grad[ref][s:d, s:d])
        dst[1] -= translation(grad[i][-d:-s, s:d], grad[ref][-d:-s, s:d])
        dst[2] -= translation(grad[i][s:d, -d:-s], grad[ref][s:d, -d:-s])
        dst[3] -= translation(grad[i][-d:-s, -d:-s], grad[ref][-d:-s, -d:-s])
        
        M = cv2.getPerspectiveTransform(src, dst)
        bbox.append(get_perspective_min_bbox(M, loaded[ref]))
        loaded[i] = cv2.warpPerspective(loaded[i], M, dsize)
    pass
    
    return loaded, np.array(bbox)
    
    
def read_tiff(fname):
    if not os.path.exists(fname): return None
    # skep detection if the corresponding output exist
    #if os.path.exists(csv): continue
    
    geotiff = rasterio.open(fname)
    data = geotiff.read()
    tr = geotiff.transform
    
    return data[0]
    
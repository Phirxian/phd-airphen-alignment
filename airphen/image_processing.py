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
    
def gaussian_kernel_1d(sigma, order, radius):
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(p(x), dtype=np.double)
    phi_x /= phi_x.sum()
    if order > 0:
        q = np.polynomial.Polynomial([1])
        p_deriv = p.deriv()
        for _ in range(order):
            q = q.deriv() + q * p_deriv
        phi_x *= q(x)
    return phi_x[np.newaxis,:]

def build_gradient(img, scale = 0.15, delta=0, method='Scharr'):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    if method == 'Sobel':
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad = gradient_normalize(grad, q=0.001)
    elif method == 'Gauss':
        sigma = 1
        sze = np.fix(6*sigma);
        gauss = gaussian_kernel_1d(sigma, 0, np.int(sze)//2);
        second = gaussian_kernel_1d(sigma, 2, np.int(sze)//2);
        Gxx = (gauss.T * second)#[:, :, np.newaxis]
        Gyy = (gauss * second.T)#[:, :, np.newaxis]
        grad_x = cv2.filter2D(img, -1, Gxx)
        grad_y = cv2.filter2D(img, -1, Gyy)
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
        
    rotation_scale = S.data['rotation-matrix']
    translation_model = S.data['curve-fit']
    
    for i in range(len(loaded)):
        x = eval_model(S.height, *translation_model[i, :4])
        y = eval_model(S.height, *translation_model[i, 4:])
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
    
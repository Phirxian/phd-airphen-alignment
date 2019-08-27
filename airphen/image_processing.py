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
    if G is None:
        raise ValueError('image or blur is None')
    return i/(G+1)*255
pass

def false_color_normalize(i):
    s = math.ceil(i.shape[0]**0.4) // 2 * 2 +1
    G = cv2.GaussianBlur(i,(s,s),cv2.BORDER_DEFAULT)
    if G is None:
        raise ValueError('image or blur is None')
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

def affine_transform_linear(S, loaded):
    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    transform = [None] * len(loaded)
    
    def eval_model(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
        
    # mean for factor $a,b,c,d$
    rotation_scale = [
       [ 1.00338690e+00, -2.36790400e-04,  7.42199784e-04,  1.00322241e+00],
       [ 9.96668519e-01, -6.35845767e-04,  1.33195202e-03,  9.97611207e-01],
       [ 1.00287955e+00, -1.44683277e-04, -2.50432200e-04,  1.00280779e+00],
       [ 9.98396830e-01,  2.36257574e-03, -1.61286382e-03,  9.97841848e-01],
       [ 9.97775109e-01, -4.42514896e-04,  2.19043135e-04,  9.96903476e-01],
       [ 1.00091705e+00, -9.09167955e-04, -4.29086670e-04,  1.00163444e+00],
       #[1.0036159  0.99666875 1.00269805 0.9988595  0.99782515 1.0003496 ]
       #[-4.70739195e-04 -7.48567882e-04  9.36510863e-05  2.09068802e-03 -4.99350475e-04 -4.66122814e-04]
       #[ 0.00074741  0.0011211  -0.00017564 -0.00169793  0.00027161 -0.00026639]
       #[1.00329896 0.9969865  1.00280747 0.99888439 0.99769919 1.00034554]
    ]
    
    translation_model_params = [
        ([ -0.214341  ,   3.0647655 , -15.68355805,  26.53635286], [ 4.89009072e-02, -5.43008533e-01,  1.81761102e+00,   -5.09998059e+01]),
        ([ -0.29732821,   3.78134406, -17.44772836,  35.72751695], [ -0.40024226   ,  5.23050824    , -24.75862376   ,   79.10072998    ]),
        ([  0.28576632,  -3.75413829,  17.67938069, -24.27180615], [ 1.02607225e-02, -5.42365651e-02, -4.57016171e-02,    3.07288625e+01]),
        ([ -0.30396441,   3.92977251, -18.21826028,  37.60667257], [  0.38182383   , -5.05192214    ,  24.18969389   , -106.39507804    ]),
        ([  0.28246774,  -3.64513373,  17.13176686, -48.95685931], [  0.34669138   , -4.67803776    ,  23.00677902   ,  -23.31192197    ]),
        ([  0.27330859,  -3.63229474,  17.37849686, -27.71094479], [ -0.41024805   ,  5.34334935    , -25.10841229   ,   71.80935994    ]),
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
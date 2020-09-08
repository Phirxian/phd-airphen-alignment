import cv2

from .data import *
from .multiple_refinement import *
from .image_processing import *

def pair_band_iterator(k):
    for i in range(k):
        for j in range(k):
            if i == j and i<j: continue
            yield (i,j)

class SpectralImage:
    def __init__(self, set, subset, prefix, height):
        self.set = set
        self.subset = subset
        self.prefix = prefix
        self.path = set + '/' + str(subset) + '/'
        self.loaded = [read_tiff(self.path + prefix + str(i) + 'nm.tif') for i in all_bands]
        
        self.ground_thrust = cv2.imread(self.path + prefix + 'mask.jpg')
        
        if self.ground_thrust is None:
            self.ground_thrust = cv2.imread(self.path + prefix + 'mask.png')
        
        if self.ground_thrust is None:
            self.ground_thrust = np.zeros((self.loaded[0].shape[0], self.loaded[0].shape[1], 3))
        
        # nearest chessboard points if not given
        if type(height) is str:
            self.chessboard = np.load(height).astype('float32')
        self.registred = self.loaded
        self.height = height
    pass
    
    def undistort(self):
        try:
            mtx = np.load(self.set + 'len_mtx.npy')
            dist = np.load(self.set + 'len_dist.npy')
            cam = np.load(self.set + 'len_cameramtx.npy')
        except:
            print('loading undist mtx error')
            return self.registred
            
        self.registred = [
            cv2.undistort(i, mtx, dist, None, cam)
            for i in self.registred
        ]
        
        self.ground_thrust = cv2.undistort(self.ground_thrust, mtx, dist, None, cam)
        
        return self.registred
    pass
    
    def spectral_registration(self, method='GFTT', reference=1, verbose=0):
        self.registred = self.loaded
        nb_kp = 0

        if verbose > 0:
            print('affine correction ...')
            
        def eval_inv_model(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d
            
        inv_model = np.array([
            [-0.00424059    ,  0.06992566    , -0.52159131    ,  3.53416035],
            [-2.21715358e-03,  1.04083738e-01, -1.75671654e+00,  1.21939587e+01],
            [0.00377931     ,  0.02339037    ,  0.18219811    ,  2.27086661],
            [-3.45198124e-03,  1.56572048e-01, -2.49924103e+00,  1.58468578e+01],
            [2.26938044e-03 ,  1.94965085e-01,  5.72026284e+00,  5.89777906e+01],
            [0.00356177     ,  0.05984146    ,  0.47140347    ,  3.37161168],
        ])
            
        # alligne trough affine transfrom using pre-callibration
        if type(self.height) is str: self.registred, transform = affine_transform(self, self.registred)
        else: self.registred, transform = affine_transform_linear(self, self.registred)
        max_xy = np.min(transform[:,:,2], axis=0).astype('int')
        min_xy = np.max(transform[:,:,2], axis=0).astype('int')
        crop_all(self, self.registred, np.flip(min_xy), np.flip(max_xy))
        
        # refine allignment with homography
        if method is None:
            pass
        elif method == 'FFT':
            if verbose > 0:
                print('fft similarity based correction ...')
            self.registred, bbox = perspective_similarity_transform(self, self.registred, reference)
            min_xy = np.max(bbox[:, :2], axis=0).astype('int')
            max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
            crop_all(self, self.registred, min_xy, max_xy)
        else:
            if verbose > 0:
                print('homography correction ...')
            
            if reference == None:
                #reference, iterator = 5, [(4,5), (3,4), (2,4), (0,2), (1,2)]
                reference, iterator = 3, [(2,3), (1,2), (5,1), (0,5), (4,5)]
            elif reference < 0:
                reference = abs(reference)-1
                iterator = pair_band_iterator(len(self.registred))
            else:
                iterator = [(i, reference) for i in range(len(self.registred))]
                
            self.registred, bbox, nb_kp, centers = multiple_refine_allignement(
                self.subset+'-', self.registred,
                method, iterator, reference, verbose
            )
            
            min_xy = np.max(bbox[:, :2], axis=0).astype('int')
            max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
            crop_all(self, self.registred, min_xy, max_xy)
            
            all_translation = transform[:,:,2] - centers
            estimated = np.array([eval_inv_model(all_translation[i,0], *inv_model[i]) for i in range(len(all_bands))])
            self.estimated_height = estimated.mean()
            
            if verbose > 0:
                print('re-estimated height =', estimated.mean())
        pass
        
        return self.registred, nb_kp
    pass

    def compute_false_color(self):
        img = np.zeros((self.registred[0].shape[0], self.registred[0].shape[1], 3))
        #img[:,:,0] = false_color_normalize(self.registred[0])*92  # B
        #img[:,:,1] = false_color_normalize(self.registred[1])*220 # G
        #img[:,:,2] = false_color_normalize(self.registred[2])*200 # R
        img[:,:,0] = 40+false_color_normalize(self.registred[0])*92  # B
        img[:,:,1] = 40+false_color_normalize(self.registred[1])*220 # G
        img[:,:,2] = 40+false_color_normalize(self.registred[2])*200 # R
        return img.clip(0,255).astype('uint8')
    pass
pass
import cv2

from .data import *
from .refinement import *
from .image_processing import *

class SpectralImage:
    def __init__(self, set, subset, prefix, height):
        self.path = set + '/' + str(subset) + '/'
        self.loaded = [read_tiff(self.path + prefix + str(i) + 'nm.tif') for i in all_bands]
        
        self.ground_thrust = cv2.imread(self.path + prefix + 'mask.jpg')
        
        if self.ground_thrust is None:
            self.ground_thrust = np.zeros((self.loaded[0].shape[0], self.loaded[0].shape[1], 3))
        
        # nearest chessboard points if not given
        self.chessboard = np.load(height).astype('float32')
        self.registred = self.loaded
    pass
    
    def spectral_registration(self, method='GFTT', reference=1, verbose=0):
        self.registred = self.loaded
        nb_kp = self.chessboard.size

        if verbose > 0:
            print('affine correction ...')
            
        # alligne trough affine transfrom using pre-callibration
        self.registred, transform = affine_transform(self, self.registred)
        max_xy = np.min(transform[:,:,2], axis=0).astype('int')
        min_xy = np.max(transform[:,:,2], axis=0).astype('int')
        crop_all(self, self.registred, np.flip(min_xy), np.flip(max_xy))
        
        # refine allignment with homography
        if method is not None:
            if verbose > 0:
                print('homography correction ...')
            self.registred, bbox, nb_kp = refine_allignement(self.registred, method, reference, verbose)
            min_xy = np.max(bbox[:, :2], axis=0).astype('int')
            max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
            crop_all(self, self.registred, min_xy, max_xy)
        pass
        
        return self.registred, nb_kp
    pass

    def compute_false_color(self):
        img = np.zeros((self.registred[0].shape[0], self.registred[0].shape[1], 3))
        img[:,:,0] = false_color_normalize(self.registred[0])*92  # B
        img[:,:,1] = false_color_normalize(self.registred[1])*220 # G
        img[:,:,2] = false_color_normalize(self.registred[2])*200 # R
        return img.astype('uint8')
    pass
pass
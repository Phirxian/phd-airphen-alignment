import cv2
import pickle as pkl

from .data import *
from .multiple_refinement import *
from .image_processing import *

def pair_band_iterator(k):
    for i in range(k):
        for j in range(k):
            if i == j and i<j: continue
            yield (i,j)

class SpectralImage:
    def __init__(self, set, subset, prefix, config, height):
        self.set = set
        self.subset = subset
        self.prefix = prefix
        self.path = set + '/' + str(subset) + '/'
        
        self.loaded = [None] * len(all_bands)
        self.mtx = [None] * len(all_bands)
        self.dist = [None] * len(all_bands)
        self.cameramtx = [None] * len(all_bands)
        
        self.data = pkl.load(open(config, 'rb'))
        self.inv_model = self.data['curve-fit-inv-x']
        
        try:
            for i,b in enumerate(all_bands):
                self.mtx[i] = self.data['len_mtx_' + str(b)]
                self.dist[i] = self.data['len_dist_' + str(b)]
                self.cameramtx[i] = self.data['len_cameramtx_' + str(b)]
                self.loaded[i] = read_tiff(self.path + prefix + str(b) + 'nm.tif')
                self.loaded[i] = cv2.undistort(self.loaded[i], self.mtx[i], self.dist[i], None, self.cameramtx[i])
        except:
            raise FileNotFoundError()
    
        self.mtx = np.vstack(self.mtx)
        self.cameramtx = np.vstack(self.mtx)

        self.ground_thrust = cv2.imread(self.path + prefix + 'mask.jpg')
        
        if self.ground_thrust is None:
            self.ground_thrust = cv2.imread(self.path + prefix + 'mask.png')
        
        if self.ground_thrust is None:
            self.ground_thrust = np.zeros((self.loaded[0].shape[0], self.loaded[0].shape[1], 3))
        
        # nearest chessboard points if not given
        self.chessboard = self.data[str(height)].astype('float32')
        self.registred = self.loaded
        self.height = height
    pass
    
    
    def spectral_registration(self, config):
        self.registred = self.loaded
        nb_kp = 0
        
        method = config.get('method','GFTT')
        reference = config.get('reference',1)
        verbose = config.get('verbose',0)

        if verbose > 0:
            print('affine correction ...')
            
        def eval_inv_model(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d
        
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
                sink, iterator = 3, [(2,3), (1,2), (5,1), (0,5), (4,5)]
            elif reference < 0:
                sink = abs(reference)-1
                iterator = pair_band_iterator(len(self.registred))
            else:
                iterator = [(i, reference) for i in range(len(self.registred))]
                sink = reference
                
            self.registred, bbox, nb_kp, centers = multiple_refine_allignement(
                self.registred, config, iterator, sink
            )
            
            min_xy = np.max(bbox[:, :2], axis=0).astype('int')
            max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
            crop_all(self, self.registred, min_xy, max_xy)
            
            all_translation = transform[:,:,2] - centers
            estimated_height = [eval_inv_model(all_translation[i,0], *self.inv_model[i]) for i in range(len(self.inv_model)) if i != reference]
            estimated_height = np.sort(estimated_height)
            self.estimated_height = estimated_height[1]
            
            if verbose > 0:
                print('re-estimated height =', estimated_height)
                
        if config.get('auto-crop-resize', False):
            mtx = np.eye(3)
            dist = np.zeros((1,4))
            cammtx = np.eye(3)
            cammtx[0,2] = -21
            cammtx[1,2] = -21
            cammtx[0,0] = 1.2
            cammtx[1,1] = 1.2
            
            for i in range(len(self.registred)):
                self.registred[i] = self.registred[i][21:-21:, 21:-21]
                self.registred[i] = cv2.resize(self.registred[i], (1200,800))
        
        return self.registred, nb_kp
        

    def compute_false_color(self):
        img = np.zeros((self.registred[0].shape[0], self.registred[0].shape[1], 3))
        #img[:,:,0] = false_color_normalize(self.registred[0])*92  # B
        #img[:,:,1] = false_color_normalize(self.registred[1])*220 # G
        #img[:,:,2] = false_color_normalize(self.registred[2])*200 # R
        img[:,:,0] = 20+false_color_normalize(self.registred[0])*92  # B
        img[:,:,1] = 20+false_color_normalize(self.registred[1])*220 # G
        img[:,:,2] = 20+false_color_normalize(self.registred[2])*200 # R
        return img.clip(0,255).astype('uint8')
        
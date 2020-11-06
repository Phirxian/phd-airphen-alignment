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
    def __init__(self, set, subset, prefix, conf_dir, height):
        self.set = set
        self.subset = subset
        self.prefix = prefix
        self.path = set + '/' + str(subset) + '/'
        
        self.loaded = [None] * len(all_bands)
        self.mtx = [None] * len(all_bands)
        self.dist = [None] * len(all_bands)
        self.cameramtx = [None] * len(all_bands)
        
        pad = 22
        
        try:
            for i,b in enumerate(all_bands):
                self.mtx[i] = np.load(conf_dir+'len_mtx_' + str(b) + '.npy')
                self.dist[i] = np.load(conf_dir+'len_dist_' + str(b) + '.npy')
                self.cameramtx[i] = np.load(conf_dir+'len_cameramtx_' + str(b) + '.npy')
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
        if type(height) is str:
            self.chessboard = np.load(conf_dir+height).astype('float32')
        self.registred = self.loaded
        self.height = height
    pass
    
    
    def spectral_registration(self, method='GFTT', reference=1, verbose=0):
        self.registred = self.loaded
        nb_kp = 0

        if verbose > 0:
            print('affine correction ...')
            
        def eval_inv_model(x, a, b, c, d):
            return a*x**3 + b*x**2 + c*x + d
            
        inv_model = np.array([
            [ 9.84929496e-04, -1.10817777e-02, -6.89656283e-01,  1.21915557e+01],
            [ 8.11305726e-04,  2.14195701e-02, -5.69608997e-01,  4.60486713e+00],
            [-1.97690865e-03, -3.43788299e-02,  3.78337961e-01,  8.63069926e+00],
            [ 4.17226398e-03, -2.94070959e-01,  6.39283208e+00, -3.78866604e+01],
            [-3.79161051e-03, -1.80322884e-01, -2.39035620e+00, -3.84722023e+00],
            [-1.64459040e-03, -4.65500887e-02,  1.18723799e-01,  9.67064868e+00],
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
                self.registred, method, iterator, reference, verbose
            )
            
            min_xy = np.max(bbox[:, :2], axis=0).astype('int')
            max_xy = np.min(bbox[:, 2:], axis=0).astype('int')
            crop_all(self, self.registred, min_xy, max_xy)
            
            all_translation = transform[:,:,2] - centers
            estimated_height = [eval_inv_model(all_translation[i,0], *inv_model[i]) for i in range(len(inv_model)) if i != reference]
            estimated_height = np.sort(estimated_height)
            self.estimated_height = estimated_height[1]
            
            if verbose > 0:
                print('re-estimated height =', estimated_height)
                
        #mtx = np.eye(3)
        #dist = np.zeros((1,4))
        #cammtx = np.eye(3)
        #cammtx[0,2] = -21
        #cammtx[1,2] = -21
        #cammtx[0,0] = 1.2
        #cammtx[1,1] = 1.2
        #
        #for i in range(len(self.registred)):
            #self.registred[i] = self.registred[i][21:-21:, 21:-21]
            #self.registred[i] = cv2.resize(self.registred[i], (1200,800))
        
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
        
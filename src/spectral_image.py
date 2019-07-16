import rasterio
import cv2
import os

from .data import *

class SpectralImage:
    def __init__(self, set, subset, prefix='', height=None):
        path = set + '/' + str(subset) + '/'
        self.loaded = [self.read_tiff(path + prefix + str(i) + 'nm.tif') for i in all_bands]
        
        self.ground_thrust = cv2.imread(path + 'mask.jpg')
        
        if self.ground_thrust is None:
            self.ground_thrust = np.zeros((self.loaded[0].shape[0], self.loaded[0].shape[1], 3))
        
        # nearest chessboard points if not given
        self.chessboard = np.load('./data/'+(subset if height == None else height)+'.npy').astype('float32')
    pass
    
    def read_tiff(self, fname):
        if not os.path.exists(fname): return None
        # skep detection if the corresponding output exist
        #if os.path.exists(csv): continue
        
        geotiff = rasterio.open(fname)
        data = geotiff.read()
        tr = geotiff.transform
        
        return data[0]
    pass
pass
#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from src.data import *
from src.spectral_image import *
from src.keypoint import *
from src.settings import *

for method in ['KAZE']:
    references_by_height = []
    
    print('--------------')
    print(method)
    print('--------------')
    
    for h in height:
        reference_val = []
        
        for mid in range(len(all_bands)):
            start_time = time.time()
            S = SpectralImage('./data/steep/', str(h))
            loaded, nb_kp = load_spectral_bands(
                S, method=method, reference=mid
            )
            reference_val.append(nb_kp + [(time.time() - start_time)])
        pass
        
        references_by_height.append(reference_val)
        print(h, reference_val)
    pass

    references_by_height = np.array(references_by_height)
    np.save('figures/keypoint-reference-count-'+method+'.npy', references_by_height)
pass
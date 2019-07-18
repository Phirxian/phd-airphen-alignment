#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import time

from tqdm import tqdm
from airphen.data import *
from airphen.spectral_image import *

def extract_performances(m):
    print('--------------')
    print(m)
    print('--------------')
    
    references_by_height = []
    
    for h in height:
        reference_val = []
        
        for mid in range(len(all_bands)):
            start_time = time.time()
            S = SpectralImage('./data/steep/', str(h), '', './data/'+str(h)+'.npy')
            loaded, nb_kp = S.spectral_registration(m, mid)
            reference_val.append(nb_kp + [(time.time() - start_time)])
        pass
        
        references_by_height.append(reference_val)
        print(h, reference_val)
    pass

    references_by_height = np.array(references_by_height)
    
    np.save('figures/keypoint-reference-count-'+m+'.npy', references_by_height)
pass

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--methods", type=str, default=all_methods,
        help="methods to extract", nargs='+',
        choices=all_methods
    )
    
    args = parser.parse_args()

    for m in args.methods:
        extract_performances(m)
pass
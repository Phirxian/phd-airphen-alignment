#!/usr/bin/python3
import numpy as np

height = np.array([
    #1.0, 1.2, 1.4
    1.6, 1.8, 2.0,
    2.2, 2.4, 2.6,
    2.8, 3.0, 3.2,
    3.4, 3.6, 3.8,
    4.0, 4.2, 4.4,
    4.6, 4.8, 5.0
])

# excluded green (reference)
bands_text = ['blue', 'green', 'red', 'redge', 'redge-max', 'nir']
all_bands = [450, 570, 675, 710, 730, 850]

all_methods = ['ORB', 'AKAZE', 'KAZE', 'BRISK', 'AGAST', 'MSER', 'SURF', 'FAST', 'GFTT']
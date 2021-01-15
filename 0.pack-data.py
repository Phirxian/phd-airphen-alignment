#!/usr/bin/python3
import numpy as np
import pickle as pkl
from glob import glob

fname = glob('data/*.npy')
config = {}

for f in fname:
    key = f.split('/')[1]
    key = key[:-4]
    print('pack', f, ' -> ', key)
    value = np.load(f)
    config[key] = value
    
with open('data/config.pkl', "wb") as output_file:
    pkl.dump(config, output_file)
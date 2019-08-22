#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from airphen.data import *

for method in all_methods:
    data = np.load('figures/keypoint-reference-count-'+method+'.npy', allow_pickle=True)
    data = data[:,:,:6]

    data = np.min(data, axis=0)
    all_min = data.astype('float')
    for i in range(data.shape[1]):
        all_min[i,i] = np.Inf
    all_min = all_min.min(axis=0)
    
    data = np.vstack([data, all_min])
    
    best = np.argmax(all_min)

    df = pd.DataFrame(
        data, index=[str(i) for i in all_bands] + ['all_min'],
        columns=pd.Index(['ref='+str(i) for i in all_bands], name='Genus')
    )
                     
    df.plot(kind='barh',figsize=(12,6))

    plt.xlim([-1,1000])
    plt.xlabel('Minimum of matched features in all images (cliped to 1000)')
    plt.ylabel('The spectral band used as reference (-1 on error)')
    plt.title(
        'Number of matches between each spectral band using ' + method + '\n' +
        'The best reference is ' + str(all_bands[best]) + ' with ' + str(np.int0(all_min[best])) + ' matches'
    )
    
    plt.legend(loc='upper right')
    plt.savefig('figures/comparaison-keypoint-matching-reference-'+method+'.png')
pass
#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.data import *

all_times = []
all_best = []

for method in all_methods:
    data = np.load('figures/keypoint-reference-count-'+method+'.npy', allow_pickle=True)
    time = data[:,:,6].mean()
    data = data[:,:,:6]

    data = np.min(data, axis=0)
    all_min = data.astype('float')
    for i in range(data.shape[1]):
        all_min[i,i] = np.Inf
    best = all_min.min(axis=0).max()
    
    all_times.append(time.flatten())
    all_best.append(best)
pass

fig, axes = plt.subplots(1, 2, figsize=(10,3))

all_times = np.array(all_times)
df = pd.DataFrame(all_times, index=all_methods,columns=pd.Index(['computation time in seconds']))
df.plot(ax=axes[0], kind='barh')
axes[0].legend(loc='lower right')

all_best = np.array(all_best)
df = pd.DataFrame(all_best, index=all_methods,columns=pd.Index(['number of features']))
df.plot(ax=axes[1], kind='barh')
axes[1].legend(loc='lower right')

plt.suptitle('Keypoint extraction performances')
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig('figures/comparaison-keypoint-performances.png')
plt.show()
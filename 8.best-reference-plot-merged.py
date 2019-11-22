#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from airphen.data import *

all_min = []
all_best = []
all_times = []

for method in all_methods:
    data = np.load('figures/keypoint-reference-count-'+method+'.npy', allow_pickle=True)
    time = data[:,:,6].mean()
    data = data[:,:,:6]
    
    print(method + ' >> ' + str(time))

    data = np.min(data, axis=0)
    local_min = data.astype('float')
    for i in range(data.shape[1]):
        local_min[i,i] = np.Inf
    local_min = local_min.min(axis=0)
    
    all_min.append(local_min)
    all_best.append(np.max(local_min))
    all_times.append(time.mean())
pass


all_min = np.array(all_min)
all_times = np.array(all_times)
all_best = np.array(all_best)

value = all_best / all_times * 2
idx = np.argsort(value)
idx = idx[-10:]

all_min = all_min[idx]
all_times = all_times[idx]
all_best = all_best[idx]
all_methods = np.array(all_methods)[idx]

df = pd.DataFrame(
    all_min, index=all_methods,
    columns=pd.Index(['ref='+str(i) for i in all_bands], name='Genus')
)
                 
df.plot(kind='bar',figsize=(6,5))

plt.ylim([-1,600])
plt.title('Number of matches between each spectral \n for each relevant methods')
plt.legend(loc='upper right')
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig('figures/comparaison-keypoint-matching-reference-merged.png')
plt.show()
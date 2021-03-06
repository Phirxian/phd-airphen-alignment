#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from airphen.data import *

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
    
    all_times.append(time.mean())
    all_best.append(best)
pass

all_times = np.array(all_times)
all_best = np.array(all_best)
value = all_best / all_times * 2
merged = np.vstack([all_times * 10, all_best, value]).transpose()

fig, axes = plt.subplots(1, 1, figsize=(10,5))
columns = ['10x time in seconds', 'number of matches', 'matches/time']

df = pd.DataFrame(
    merged, index=all_methods,
    columns=pd.Index(columns)
)

df = df.sort_values(['matches/time'])
#df = df.sort_values([columns[0]])
df.plot(ax=axes, kind='bar')
plt.ylim([0,500])
plt.yticks(np.arange(0,550,50))

plt.xlabel('Methodes and modalities')
plt.ylabel('Performances in different terms (time, matches, ...)')

plt.suptitle('Keypoint extraction performances')
plt.tight_layout(rect=(0,0,1,0.97))
plt.savefig('figures/comparaison-keypoint-performances.png')
plt.show()
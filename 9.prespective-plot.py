#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from airphen.data import *

with open('figures/prespective-error.txt') as f:
    content = f.readlines()

data = [None] * len(content)
    
for id,l in enumerate(content):
    l = l.split(';')
    l = [i.split('=') for i in l]
    l = [float(i[1]) for i in l]
    data[id] = l
pass

data = np.array(data)
    
plt.figure(figsize=(9,2.5))
axes = plt.gca()
axes.set_ylim([0.5,1.5])

bands = all_bands.copy()
bands.remove(570) # the reference

for i,b in enumerate(bands):
    l2 = data[i::5,1]
    std = data[i::5,2]/2
    x = np.arange(len(l2))
    #c = all_bands.index(b)
    plt.plot(x, l2, label=str(bands[i]))
    #plt.fill_between(x, l2-std, l2+std, alpha=.2)
pass
    
plt.title('Allignement error at different height')
plt.xlabel('height of the aquisition')
plt.ylabel('L2 distance in pixel (error)')
plt.xticks(np.arange(height.shape[0]), height)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('figures/prespective-allignement-rmse.jpg')
plt.show()
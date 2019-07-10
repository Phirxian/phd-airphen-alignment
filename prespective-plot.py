#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

from src.data import *

with open('tmp/prespective-error.txt') as f:
    content = f.readlines()

data = [None] * len(content)
    
for id,l in enumerate(content):
    l = l.split(';')
    l = [i.split('=') for i in l]
    l = [float(i[1]) for i in l]
    data[id] = l
pass

data = np.array(data)
    
plt.figure(figsize=(15,4))
axes = plt.gca()
axes.set_ylim([0,2])

for i in range(len(bands)):
    rmse = data[i::5,1]
    std = data[i::5,2]
    plt.plot(rmse, label=bands_text[i])
pass
    
plt.title('Allignement error at different height')
plt.xlabel('height of the aquisition')
plt.ylabel('L2 distance in pixel (error)')
plt.xticks(np.arange(height.shape[0]), height)
plt.legend()
plt.tight_layout()
plt.savefig('tmp/prespective-allignement-rmse.jpg')
plt.show()
#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import cv2

from airphen import *

all_transform_x = []
all_transform_y = []

for i in height:
    chessboard = np.load('data/' + str(i) + '.npy')
    centroid = chessboard[:,:,0,:].mean(axis=0).astype('float32') 
    transform = [None] * len(all_bands)
    
    for b in range(len(all_bands)):
        points = chessboard[b,:,0,:]
        if cv2.__version__[0] == '4': transform[b] = cv2.estimateAffine2D(points, centroid)[0]
        else: transform[b] = cv2.estimateRigidTransform(points, centroid, fullAffine=False)
        transform[b] = transform[b][0:2, 2].reshape(-1)
    pass
    
    transform = np.array(transform)
    all_transform_x.append(transform[:,0])
    all_transform_y.append(transform[:,1])
pass

all_transform_x = np.array(all_transform_x)
all_transform_y = np.array(all_transform_y)

leg = np.array(bands_text)

plt.figure(figsize=(10,4))

plt.subplot(121)
for i in range(all_transform_x.shape[1]):
    plt.plot(height, all_transform_x[:,i], label=leg[i])
plt.xlabel('distance from ground in meter')
plt.ylabel('translation x in pixel')
    
plt.subplot(122)
for i in range(all_transform_y.shape[1]):
    plt.plot(height, all_transform_y[:,i], label=leg[i])
plt.xlabel('distance from ground in meter')
plt.ylabel('translation y in pixel')

plt.suptitle('Initial affine translation for each height')
plt.legend()
#plt.tight_layout()

plt.savefig('figures/affine-translation-height.png')
plt.show()
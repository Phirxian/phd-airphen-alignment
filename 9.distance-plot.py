#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import cv2

from airphen import *

#######################################################

all_transform_a = []
all_transform_b = []
all_transform_c = []
all_transform_d = []
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
        #print(transform[b])
    pass
    
    transform = np.array(transform)
    print(transform)
    
    all_transform_a.append(transform[:,0,0])
    all_transform_b.append(transform[:,0,1])
    all_transform_c.append(transform[:,1,0])
    all_transform_d.append(transform[:,1,1])
    all_transform_x.append(transform[:,0,2])
    all_transform_y.append(transform[:,1,2])
pass

all_transform_a = np.array(all_transform_a)
all_transform_b = np.array(all_transform_b)
all_transform_c = np.array(all_transform_c)
all_transform_d = np.array(all_transform_d)
all_transform_x = np.array(all_transform_x)
all_transform_y = np.array(all_transform_y)

leg = np.array([str(i) for i in all_bands])
#leg = np.array(bands_text)

#######################################################
plt.figure(figsize=(7,5))
#######################################################

plt.subplot(221)
plt.title('Affine matrix factor A')
for i in range(all_transform_a.shape[1]):
    plt.scatter(height, all_transform_a[:,i], s=4, c='black')
    plt.plot(height, all_transform_a[:,i], label=leg[i])
    
plt.ylabel('translation in pixel')
    
plt.subplot(222)
plt.title('Affine matrix factor B')
for i in range(all_transform_a.shape[1]):
    plt.scatter(height, all_transform_b[:,i], s=4, c='black')
    plt.plot(height, all_transform_b[:,i], label=leg[i])
    
plt.subplot(223)
plt.title('Affine matrix factor C')
for i in range(all_transform_a.shape[1]):
    plt.scatter(height, all_transform_c[:,i], s=4, c='black')
    plt.plot(height, all_transform_c[:,i], label=leg[i])
    
plt.ylabel('translation in pixel')
plt.xlabel('height in meter')
    
plt.subplot(224)
plt.title('Affine matrix factor D')
for i in range(all_transform_a.shape[1]):
    plt.scatter(height, all_transform_d[:,i], s=4, c='black')
    plt.plot(height, all_transform_d[:,i], label=leg[i])
    
plt.xlabel('height in meter')
    
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('figures/affine-rotation-height.png')

#######################################################
plt.figure(figsize=(7,3))
#######################################################

plt.subplot(121)
plt.title('Affine matrix factor X')
for i in range(all_transform_a.shape[1]):
    plt.scatter(height, all_transform_x[:,i], s=4, c='black')
    plt.plot(height, all_transform_x[:,i], label=leg[i])
plt.ylabel('translation in pixel')
plt.xlabel('height in meter')
    
plt.subplot(122)
plt.title('Affine matrix factor Y')
for i in range(all_transform_a.shape[1]):
    plt.scatter(height, all_transform_y[:,i], s=4, c='black')
    plt.plot(height, all_transform_y[:,i], label=leg[i])

plt.xlabel('height in meter')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('figures/affine-translation-height.png')

plt.show()
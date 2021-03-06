#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.optimize import curve_fit
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
leg = np.array(bands_text)

#######################################################

print('a =', all_transform_a.mean(axis=0))
print('b =', all_transform_b.mean(axis=0))
print('c =', all_transform_c.mean(axis=0))
print('d =', all_transform_d.mean(axis=0))

#######################################################

def func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

fitted_x = []
fitted_y = []

plt.figure(figsize=(10,4))
plt.subplot(121)

plt.title('fitted curve : translation x')
for i in range(all_transform_x.shape[1]):
    x, y = height, all_transform_x[:,i]
    popt, pcov = curve_fit(func, x, y)
    x_resampled = np.arange(min(x), max(x), 0.0025)
    plt.scatter(x, y, s=4, c='black', label=bands_text[i])
    plt.plot(x_resampled, func(x_resampled, *popt), label=bands_text[i])
    print(bands_text[i], 'x :', popt)
    
plt.subplot(122)

plt.title('fitted curve : translation y')
for i in range(all_transform_y.shape[1]):
    x, y = height, all_transform_y[:,i]
    popt, pcov = curve_fit(func, x, y)
    x_resampled = np.arange(min(x), max(x), 0.0025)
    plt.scatter(x, y, s=4, c='black', label=bands_text[i])
    plt.plot(x_resampled, func(x_resampled, *popt), label=bands_text[i])
    print(bands_text[i], 'y :', popt)
    
plt.savefig('figures/affine-curve-fit.png')
plt.show()
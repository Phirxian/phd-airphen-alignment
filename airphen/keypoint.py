#!/usr/bin/python3
import numpy as np
import itertools
import math

from scipy.spatial import distance

class FilterDetection:

    def __init__(self, matches, kp1, kp2):
        self.matches = matches
        self.kp1 = kp1
        self.kp2 = kp2
        self.a = np.array([self.kp1[i.queryIdx].pt for i in self.matches], dtype=float)
        self.b = np.array([self.kp2[i.trainIdx].pt for i in self.matches], dtype=float)
    pass

    def filter_distance(self, t):
        #print('filter distances ...')
        return np.array([
            i for i in range(len(self.a))
            if self.matches[i].distance < t
        ])
    pass

    def filter_position(self, t):
        #print('filter positions ...')
        d = np.sum((self.a-self.b)**2, axis=1)

        d = np.array([
            abs(distance.euclidean(self.a[i], [0,0]) - distance.euclidean([0,0], self.b[i]))
            for i in range(len(self.a))
        ])

        return d<(t**2)
    pass

    def filter_duplicated(self, t):
        #print('filter duplicated ...')
        d = np.array([
            min([
                distance.euclidean(self.a[i], self.a[j])
                for j in range(i)
            ] + [float("inf")])
            for i in range(len(self.a))
        ])

        return d>t
    pass

    def filter_angle(self, t):
        #print('filter angles ...')
        d = 180/math.pi*np.arctan(
            (self.a[:,1] - self.b[:,1])/
            (self.b[:,0] - self.a[:,0] + 1280*2)
        )

        return abs(d)<t
    pass

pass

def keypoint_filter(matches, kp1, kp2, alpha):
    filtering = FilterDetection(matches, kp1, kp2)
    filter = np.where(filtering.filter_angle(1))
    matches = np.array(matches)[filter]

    filtering = FilterDetection(matches, kp1, kp2)
    filter = np.where(filtering.filter_position(alpha))
    matches = np.array(matches)[filter]

    #filtering = FilterDetection(matches, kp1, kp2)
    #filter = np.where(filtering.filter_duplicated(alpha*4))
    #matches = np.array(matches)[filter]

    return matches
pass

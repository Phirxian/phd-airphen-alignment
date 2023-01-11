#!/usr/bin/python3
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from collections import namedtuple
from torch.nn.utils import clip_grad_norm_

import numpy as np
import math
import cv2

from tqdm import tqdm
from .keypoint import *
from .spanning_tree import *
from .image_processing import *
from .data import *

class Detection:
    def __init__(self,kp, des, T):
        self.kp = kp
        self.des = des
        self.T = T
        self.nb_kp = 0


def global_kp_extractor(grad, detector, descriptor, verbose):
    if verbose: print('keypoint detection')
    keypoints = []
    for i in range(0, len(grad)):
        kp = detector.detect(grad[i],None)
        kp, des = descriptor.compute(grad[i], kp)
        keypoints.append(Detection(kp, des, None))
        #cv2.imshow('grad'+str(i), grad[i])
    return keypoints

def kp_graph_matching(keypoints, descriptor, iterator, verbose, max_dist=20):
    bf = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=True)
    evaluators = [cv2.RANSAC, cv2.LMEDS, cv2.RHO]
    arcs = []

    if verbose: print('keypoint graph matching')

    for j,i in tqdm(iterator):
        if i==j:
            continue
        # kp1 = 'ref'==to kp2='reg'
        kp1, kp2 = keypoints[i].kp, keypoints[j].kp
        M = bf.match(keypoints[i].des, keypoints[j].des)
        M = keypoint_filter(M, kp1, kp2, max_dist)

        source = np.array([kp1[k.queryIdx].pt for k in M], dtype=float)
        target = np.array([kp2[k.trainIdx].pt for k in M], dtype=float)

        if True:
            source = source[np.newaxis, :].astype('int')
            target = target[np.newaxis, :].astype('int')
            T, mask = cv2.findHomography(target, source, evaluators[1])
            target = np.float32(target[0,mask,:])
            source = np.float32(source[0,mask,:])
        else:
            source = source.astype('int')
            target = target.astype('int')
            T, mask = cv2.findHomography(target, source, evaluators[1])
            mask = mask[:,0]
            target = np.float32(target[np.where(mask)])
            source = np.float32(source[np.where(mask)])

            print(T)

        print('graph', j, '->', i)
        print(source.shape)
        print(target.shape)

        """
        matched_image = cv2.drawMatches(
            np.zeros((1126, 1724, 3), dtype='uint8'), kp1,
            np.zeros((1126, 1724, 3), dtype='uint8'), kp2,
            M, None, flags=2
        )
        """

        #cv2.imshow('matched_image', matched_image)
        #cv2.waitKey()

        if T is not None:
            corrected = cv2.perspectiveTransform(target,T)
            diff = corrected-source
            l2 = np.sqrt((diff**2).sum(axis=1))
            keypoints[j].T = T
            keypoints[j].nb_kp = len(source)
            keypoints[j].weight = l2.mean()
            A = Arc(tail=j, head=i, weight=keypoints[j].weight)
            arcs.append(A)

    return arcs

class CameraModel(nn.Module):

    def __init__(self, M=np.eye(3)):
        super(CameraModel, self).__init__()
        # values from previous experiment
        self.K = torch.nn.Parameter(torch.tensor([0.069, 0.004, 0.013], dtype=torch.float32))
        self.S = torch.nn.Parameter(torch.tensor([0.008, 0.015,-0.011, 0.036], dtype=torch.float32))
        self.P = torch.nn.Parameter(torch.tensor([0.005,-0.017], dtype=torch.float32))
        #self.K = torch.nn.Parameter(torch.rand(3, dtype=torch.float32)*0.1)
        #self.P = torch.nn.Parameter(torch.rand(2, dtype=torch.float32)*0.1)
        #self.S = torch.nn.Parameter(torch.rand(4, dtype=torch.float32)*0.1)
        self.M = torch.nn.Parameter(torch.tensor(M, dtype=torch.float32))
        self.T = torch.nn.Parameter(torch.tensor(np.eye(3), dtype=torch.float32))
        self.fc = torch.nn.Linear(4*2, 2)

    def radial(self, r, xy):
        R = torch.stack([r, r**2, r**3], dim=1)
        K = self.K.repeat(len(r), 1)
        return xy * torch.sum(K * R, dim=1).unsqueeze(1)

    def tangential(self, r, xy):
        P = self.P.unsqueeze(0).repeat(len(xy), 1)
        px = torch.prod(xy, dim=1).unsqueeze(1)
        t1 = 2*P*px
        t2 = P.flip(1)*(r.unsqueeze(1)+xy**2)
        return t1 + t2

    def prismatic(self, r, xy):
        s1, s2, s3, s4 = self.S
        S1 = self.S[0] * r + self.S[1] * r**2
        S2 = self.S[2] * r + self.S[3] * r**2
        P = torch.stack([S1,S2], dim=1)
        return P

    def transform(self, T, xy):
        z = torch.ones((len(xy),len(T)-2)).to(device=xy.device)
        xy = torch.cat([xy, z], dim=1).to(dtype=torch.float32)
        M = T.repeat(len(xy), 1, 1)
        xy = xy.unsqueeze(1)
        xy = torch.bmm(xy, M)
        return xy[:,0,:2]

    def forward(self, xy):
        r = torch.sum(xy**2, dim=1)
        xy = self.transform(self.M, xy)

        R = self.radial(r, xy).to(device=xy.device)
        T = self.tangential(r, xy).to(device=xy.device)
        P = self.prismatic(r, xy).to(device=xy.device)

        if False:
            A = torch.cat([xy, R, T, P], dim=1).to(torch.float32)
            xy = self.fc(A)
        else:
            xy = xy + R + T + P

        return self.transform(self.T, xy)

    def forward_numpy(self, mapx, mapy):
        ax = mapx.reshape(np.prod(mapx.shape))
        ay = mapy.reshape(np.prod(mapy.shape))

        xy = np.vstack([ax, ay]).T
        xy = torch.tensor(xy, dtype=torch.float32).cuda()
        xy = self.forward(xy)
        xy = xy.cpu().detach().numpy()

        ax = xy[:,0].reshape(mapx.shape)
        ay = xy[:,1].reshape(mapy.shape)
        return ax, ay

def apply_spanning_registration(self, arcs, keypoints, descriptor, sink, loaded, verbose):
    tree = gen_spanning_arborescence(arcs, sink, 'min')
    tree = [i for i in tree.values()]
    #print(tree)

    dsize = (loaded[0].shape[1], loaded[0].shape[0])
    center = np.array(dsize)/2
    bbox, centers, keypoint_found = [None]*len(loaded), [None]*len(loaded), [None]*len(loaded)

    bbox[sink] = [0,0,dsize[1],dsize[0]]
    centers[sink] = [0,0]
    self.models = {}

    bf = cv2.BFMatcher(descriptor.defaultNorm(), crossCheck=True)

    for link in tree:
        if verbose:
            print('registration ', link.tail, link.head, link.weight, 'to sink', sink)

        c, M, e = link.tail, np.eye(3), 0
        while c != sink:
            l = [i for i in tree if i.tail==c][0]
            M = np.matmul(keypoints[c].T,M)
            e = e+keypoints[c].weight
            c = l.head

        print('M', M)

        kp1, kp2 = keypoints[link.tail].kp, keypoints[sink].kp
        G = bf.match(keypoints[link.tail].des, keypoints[sink].des)
        G = keypoint_filter(G, kp1, kp2, 20)
        source = np.array([kp1[k.queryIdx].pt for k in G], dtype=np.float32)
        target = np.array([kp2[k.trainIdx].pt for k in G], dtype=np.float32)

        if True:
            source = source.astype('int')
            target = target.astype('int')
            T, mask = cv2.findHomography(target, source, cv2.LMEDS)
            mask = mask[:,0]
            target = np.float32(target[np.where(mask)])
            source = np.float32(source[np.where(mask)])

        shape = loaded[link.tail].shape[:2]

        if False:
            source = np.hstack([source, np.zeros((len(target),1), dtype=np.float32)])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([source], [target], tuple(shape), None, None)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, shape, 1, tuple(shape))
            loaded[link.tail] = cv2.undistort(loaded[link.tail], mtx, dist, None, newcameramtx)
            bbox[link.tail] = roi
        else:
            source = torch.tensor(source/np.flip(shape)-0.5).cuda()
            target = torch.tensor(target/np.flip(shape)-0.5).cuda()
            print('source:', source.shape)
            print('target:', target.shape)

            model = CameraModel(M=M).cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-10, patience=20, factor=0.9)
            progress = tqdm(range(self.config.get('epochs', 3000)), desc='train')

            self.models[link.tail] = model

            for epochs in progress:
                optimizer.zero_grad()
                xy = model(target)
                #diff = abs(source-xy).sum(dim=1)
                diff = ((source-xy)**2).sum(dim=1).sqrt()
                # consider only best ones (remove possible false positive)
                diff = torch.sort(diff)[0]
                diff = diff[:int(len(diff)*0.99)]
                loss = diff.mean()
                loss.backward()
                scheduler.step(loss)
                optimizer.step()
                progress.set_postfix({'loss': loss.item()})
            print('loss:', loss.min(dim=0)[0], loss.max(dim=0)[0])

            output = model(target).cpu().detach().numpy()
            source = source.cpu().detach().numpy()
            target = target.cpu().detach().numpy()

            print('K', model.K.cpu().detach().numpy())
            print('S', model.S.cpu().detach().numpy())
            print('P', model.P.cpu().detach().numpy())

            print('source:', source.shape)
            print('target:', target.shape)
            print('output:', target.shape)

            if False:
                plt.figure()
                xy = (source+0.5)*np.flip(shape)
                a = (output+0.5)*np.flip(shape)
                b = (target+0.5)*np.flip(shape)
                uv = xy-a
                vu = xy-b
                plt.quiver(xy[:,0], xy[:,1], uv[:,0], uv[:,1], scale_units='xy', units='xy', scale=1.0, color='r', label='transformed')
                plt.quiver(xy[:,0], xy[:,1], vu[:,0], vu[:,1], scale_units='xy', units='xy', scale=1.0, color='g', label='expected')
                plt.show()

            mapx, mapy = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])) #, sparse=True)
            print('mapx:', mapx.min(), mapx.max())
            print('mapy:', mapy.min(), mapy.max())

            mapx = mapx/shape[1] - 0.5
            mapy = mapy/shape[0] - 0.5
            mapx, mapy = model.forward_numpy(mapx, mapy)
            mapx = (mapx+0.5)*shape[1]
            mapy = (mapy+0.5)*shape[0]

            print('mapx:', mapx.min(), mapx.max())
            print('mapy:', mapy.min(), mapy.max())

            mapx, mapy = np.float32(mapx), np.float32(mapy)
            loaded[link.tail] = cv2.remap(loaded[link.tail], mapx, mapy, cv2.INTER_LINEAR)
            bbox[link.tail] = np.array([0,0,*shape])

        centers[link.tail] = cv2.perspectiveTransform(np.array([[center]]),M)[0,0] - center
        keypoint_found[link.tail] = e

    return loaded, np.array(bbox), keypoint_found, np.array(centers)

# @ref the best reference : default=1=570 !! (maximum number of matches)
def multiple_refine_allignement(self, loaded, config, iterator, sink):
    ######################### image transformation

    img = [i.astype('float32')/255 for i in loaded]
    #img = [gradient_normalize(i, 0.0) for i in img]
    grad = [build_gradient(i, method=config.get('gradient-type', 'Ridge')) for i in img]

    self.grad = grad

    #for i,j in enumerate(grad):
    #    cv2.imshow(str(i), j)
    #cv2.waitKey()

    #for i,j in enumerate(grad):
    #    cv2.imshow(str(i), j)
    #cv2.waitKey()

    ######################### keypoint detection and description

    verbose = config.get('verbose', False)
    descriptor = cv2.ORB_create()
    #descriptor = cv2.SIFT_create()
    detector = detectors[config.get('method', 'GFTT')]()

    keypoints = global_kp_extractor(grad, detector, descriptor, verbose)
    arcs = kp_graph_matching(keypoints, descriptor, iterator, verbose)

    return apply_spanning_registration(self, arcs, keypoints, descriptor, sink, loaded, verbose)

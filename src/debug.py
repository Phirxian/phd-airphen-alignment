import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

def scatter_plot_residual(source, target, corrected):
    fig = plt.figure(figsize=(17,5))
    size = 1
    
    ################################################
    ####   ALIGNMENT REFINENMENT COMPARAISON   #####
    ################################################
    
    diff = target-source
    l2 = np.sqrt((diff**2)[0].sum(axis=1))
    
    ax = fig.add_subplot(121)
    ax.scatter(target[0,:,0], target[0,:,1], s=size)
    ax.scatter(source[0,:,0], source[0,:,1], s=size)
    ax.quiver(
        corrected[0,:,0], corrected[0,:,1],
        diff[0,:,0], diff[0,:,1],
        scale_units='xy', scale=.1,
        units='xy', width=2
    )
    #ax.set_aspect('equal', adjustable='box')
    plt.title('spatial distances of matched keypoints')
    plt.xlabel('matched l2 mean = ' + str(l2.mean()))
    plt.ylabel('arrow scale = 10')
    
    ################################################
    
    diff = corrected-source
    l2 = np.sqrt((diff**2)[0].sum(axis=1))
    residual = diff[0].mean(axis=0)
    
    ax = fig.add_subplot(122)
    ax.scatter(corrected[0,:,0], corrected[0,:,1], s=size)
    ax.scatter(source[0,:,0], source[0,:,1], s=size)
    ax.quiver(
        corrected[0,:,0], corrected[0,:,1],
        diff[0,:,0], diff[0,:,1],
        scale_units='xy', scale=0.1,
        units='xy', width=2
    )
    #ax.set_aspect('equal', adjustable='box')
    ax.yaxis.set_label_position("right")
    plt.title('residual distances after correction')
    plt.xlabel('residual l2 mean = ' + str(l2.mean()))
    plt.ylabel('residual std = ' + str(np.round(residual, 4)))
    
    ################################################
    
    plt.suptitle('Camera Alignment Refinement using keypoints')
    #plt.tight_layout()
    plt.savefig('figures/perspective-features-matching-scatter.png')
    
    ################################################
    #######   RESIDUAL ANGLE DISTRIBUTION   ########
    ################################################
    
    direction = np.arctan2(diff[0,:,1], diff[0,:,0])*180/math.pi + 180
    #bins = np.arange(0,180,10)
    #hog = HOG_histogram(direction, l2, bins).clip(0,100)
    
    fig = plt.figure(figsize=(17,10))
    steep=40
    subfig = np.arange(0,360,steep, dtype=int)
    sq = subfig.shape[0]**0.5
    p = sq*10 + sq*100
    
    for i,angle in enumerate(subfig):
        ax = fig.add_subplot(p+i+1)
        index = np.where(np.logical_and(direction >= angle, direction < angle+steep))
        ax.scatter(corrected[0,index,0], corrected[0,index,1], s=size)
        ax.scatter(source[0,index,0], source[0,index,1], s=size)
        ax.quiver(
            corrected[0,index,0], corrected[0,index,1],
            diff[0,index,0], diff[0,index,1],
            scale_units='xy', scale=0.1,
            units='xy', width=4
        )
        ax.yaxis.set_label_position("right")
        plt.title(f'vector bin [{angle}, {angle+steep}]')
        plt.ylabel('l2 mean = ' + str(np.round(l2[index].mean(),4)))
    pass
    
    plt.suptitle('Residual Angle Distribution')
    plt.savefig('figures/perspective-features-residual.png')
    
    ################################################
    
    plt.show()
pass

def draw_final_keypoint_matching(source, target, grad_mid, grad_i):
    kp1 = [cv2.KeyPoint(i[0], i[1], 0) for i in source[0]]
    kp2 = [cv2.KeyPoint(i[0], i[1], 0) for i in target[0]]
    matches = [cv2.DMatch(i,i,0) for i in range(source.shape[1])]
    kp = keypoint_draw(grad_mid, grad_i, kp1, kp2, matches)
    cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
    cv2.imshow(str(i), kp)
    cv2.waitKey(1)
pass
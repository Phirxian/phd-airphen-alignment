import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import numpy as np
import math
import cv2

def scatter_plot_residual(prefix, source, target, corrected, image_size, l):
    fig = plt.figure(figsize=(17,5))
    size = 1
    
    ################################################
    ####   ALIGNMENT REFINENMENT COMPARAISON   #####
    ################################################
    
    diff = (target-source)[0]
    l2 = np.sqrt((diff**2).sum(axis=1))
    
    ax = fig.add_subplot(121)
    ax.scatter(target[0,:,0], target[0,:,1], s=size)
    ax.scatter(source[0,:,0], source[0,:,1], s=size)
    ax.quiver(
        corrected[0,:,0], corrected[0,:,1],
        diff[:,0], diff[:,1],
        scale_units='xy', scale=.1,
        units='xy', width=2
    )
    #ax.set_aspect('equal', adjustable='box')
    plt.title('spatial distances of matched keypoints')
    plt.xlabel('matched l2 mean = ' + str(l2.mean()))
    plt.ylabel('arrow scale = 10')
    
    ################################################
    
    diff = (corrected-source)[0]
    l2 = np.sqrt((diff**2).sum(axis=1))
    
    ax = fig.add_subplot(122)
    ax.scatter(corrected[0,:,0], corrected[0,:,1], s=size)
    ax.scatter(source[0,:,0], source[0,:,1], s=size)
    ax.quiver(
        corrected[0,:,0], corrected[0,:,1],
        diff[:,0], diff[:,1],
        scale_units='xy', scale=0.1,
        units='xy', width=2
    )
    #ax.set_aspect('equal', adjustable='box')
    ax.yaxis.set_label_position("right")
    plt.title('residual distances after correction')
    plt.xlabel('residual l2 mean = ' + str(l2.mean()))
    
    ################################################
    
    plt.suptitle('Camera Alignment Refinement using keypoints')
    #plt.tight_layout()
    plt.savefig('figures/perspective-features-matching-scatter.png')
    
    ################################################
    #######   RESIDUAL ANGLE DISTRIBUTION   ########
    ################################################
    
    direction = np.arctan2(diff[:,1], diff[:,0])*180/math.pi + 180
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
            diff[index,0], diff[index,1],
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
    
    data = np.vstack([source[0,:,0], source[0,:,1], diff[:,0], diff[:,1]])
    print(data.shape)
    np.save('/home/javayss/'+prefix+str(l)+'.npy', data)
    
    #plt.figure()
    #
    #if False:
    #    gridx = np.arange(0.0, image_size[0], 2)
    #    gridy = np.arange(0.0, image_size[1], 2)
    #    cov_model = Gaussian(dim=2, len_scale=1, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
    #    OK1 = OrdinaryKriging(diff[:,0], diff[:,1], l2, cov_model)
    #    z1, ss1 = OK1.execute('grid', gridx, gridy)
    #    plt.imshow(z1 / z1.max(), origin="lower")
    #else:
    #    gp = GaussianProcessRegressor()
    #    gp.fit(X=diff, y=l2)
    #    
    #    r = np.linspace(0, 4, image_size[0]) 
    #    c = np.linspace(0, 4, image_size[1])
    #    rr, cc = np.meshgrid(r, c)
    #    rr_cc_as_cols = np.column_stack([rr.flatten(), cc.flatten()])
    #    interpolated = gp.predict(rr_cc_as_cols).reshape(image_size[:2])
    #    plt.imshow(interpolated, origin="lower")
    #
    #plt.savefig('figures/perspective-krigging-resiual.png')
    
    ################################################
    
    #plt.show()
pass

def keypoint_draw(img1, img2, kp1, kp2, matches):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for i in kp1:
        px = np.int0(i.pt)
        img1[px[1], px[0], :] = [0,0,255]
    
    for i in kp2:
        px = np.int0(i.pt)
        img2[px[1], px[0], :] = [0,0,255]
    
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
pass

def draw_final_keypoint_matching(source, target, grad_mid, grad_i):
    kp1 = [cv2.KeyPoint(i[0], i[1], 0) for i in source[0]]
    kp2 = [cv2.KeyPoint(i[0], i[1], 0) for i in target[0]]
    matches = [cv2.DMatch(i,i,0) for i in range(source.shape[1])]
    kp = keypoint_draw(grad_mid, grad_i, kp1, kp2, matches)
    cv2.namedWindow('kp match', cv2.WINDOW_NORMAL)
    cv2.imshow('kp match', kp)
    cv2.waitKey(1)
pass
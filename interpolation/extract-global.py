#!/usr/bin/python3
import numpy as np
import scipy.interpolate as interpolate

from matplotlib import pyplot as plt
from scipy.interpolate import griddata

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

height = np.array([
    #1.0, 1.2, 1.4
    1.6, 1.8, 2.0,
    2.2, 2.4, 2.6,
    2.8, 3.0, 3.2,
    3.4, 3.6, 3.8,
    4.0, 4.2, 4.4,
    4.6, 4.8, 5.0
])

def data_reduction(data):
    clustering = DBSCAN(eps=10, min_samples=1).fit(data[:,:2])
    #clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=10).fit(data[:,:2])
    cluster_labels = clustering.labels_
    label = np.unique(cluster_labels)
    data = np.vstack([np.mean(data[cluster_labels == i], axis=0) for i in label])
    data = np.vstack([data[:,0], data[:,1], np.sqrt(data[:,2]**2+data[:,3]**2) ]).transpose()
    return data
    
def plot_error(fig, xi, yi, data, method, idx):
    pts1 = data[:,:2]
    z1 = data[:,2]
    coords = (pts1[:,0], pts1[:,1])
    maps = (xi[None,:], yi[:,None])
    
    ax = fig.add_subplot(idx)
    ax.set_title("Scipy ("+method+") \n points " + str(data.shape[0]) + ' error ' + str(data[:,2].mean(axis=0)))
    model = np.nan_to_num(griddata(coords, z1, maps, method = method))
    model = model.clip(0, z1.max())
    #model = (model/model.max())**2 * z1.max()
    im = ax.contourf(xi, yi, model, levels=20, cmap="RdBu_r")
    ax.scatter(pts1[:,0], pts1[:,1], color = 'k', s = 2)
    plt.colorbar(im)
    
for k,h in tqdm(enumerate(height)):
    data = [np.load('byheight/'+str(h)+'-'+str(i)+'.npy').transpose() for i in [0,2,3,4,5]]
    data = np.vstack(data)
    #data = np.vstack([data_reduction(i) for i in data])
    data = data_reduction(data)
    
    xi, yi = np.arange(1250), np.arange(850)

    fig = plt.figure(figsize=(15,16))
    plot_error(fig, xi, yi, data, 'linear', 211)
    plot_error(fig, xi, yi, data, 'cubic', 212)
    plt.savefig('figures/all-'+str(h)+'-error.png')
    plt.close()
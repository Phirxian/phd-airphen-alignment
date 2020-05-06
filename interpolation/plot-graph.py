#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

height = np.array([
    #1.0, 1.2, 1.4
    1.6, 1.8, 2.0,
    2.2, 2.4, 2.6,
    2.8, 3.0, 3.2,
    3.4, 3.6, 3.8,
    4.0, 4.2, 4.4,
    4.6, 4.8, 5.0
])


bands_text =  ['blue',    'green',   'red',     'redge',   'redge_max', 'nir']
bands_color = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd',   '#8c564b']
all_bands = [450, 570, 675, 710, 730, 850]

bands_txt = ['450', 'ref', '650', '710', '730', '850']
values = np.load('values.npy')

plt.figure(figsize=(6,2))
axes = plt.gca()
axes.set_ylim([0.5,1.2])

for i in [0,2,3,4,5]:
    plt.plot(values[i], label=str(all_bands[i]), color=bands_color[i])
plt.plot(np.ones(len(values[0])), ':', color='gray')

axis_label = [str(x) if i%2 == 0 else '' for i,x in enumerate(height)]
    
plt.title('Perspective error at different heights')
plt.xlabel('height of the aquisition')
plt.ylabel('L2 distance in pixel')
plt.xticks(np.arange(height.shape[0]), axis_label)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig('figures/prespective-allignement-rmse.png')
plt.show()
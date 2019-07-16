# Airphen Camera Spectral Band Alignment

# Materiel and Data

![alt text](https://www.hiphen-plant.com/wp-content/webp-express/webp-images/doc-root/wp-content/uploads/2019/05/airphen-detail3.png.webp "the airphen camera")

For the calibration one shot of a chessboard is taken at different height  (0.8 to 5.0 meter with 20cm steep)
The corresponding data can by found in data/steep-chess/

For the alignment verification one shot of a grassland is taken at different height (0.8 to 5.0 meter with 20cm steep)
The corresponding data can by found in data/steep/

# Method

Allignement is refined at different stage (calibration, affine, prespective)
Each steep is explained here with corresponding figures and metrics.

## Phase 0 (Callibration):

+ detecte chessboard at different height of acquisition
+ order detected points by x/y (detection can be fliped depending of the moon position)
+ each detected point is saved on data/*.npy

## Phase 1 (Affine Correction):

![alt text](figures/math-affine-correction.png "equation of the affine correction")

+ selecte the detected points of the nearest height (know by the user or using sensor)
+ compute the centroid grid (each point mean)
+ compute affine transfrom from each spectral band to this centroid grid
+ crop each spectral bands to the minimal bbox

![alt text](figures/affine-allignement-rmse.jpg "Affine Reprojection Error")
![alt text](figures/affine_5.0_false_color.jpg "False Color Corrected Image")

## Phase 2 (Perspective Correction):

Each spectral band have different properties and value by nature,
but we can extract corresponding similarity by transforming each spectral band to it's derivative
to find similarity in gradient break of those ones. The Sharr filter are used in this steep for is precision.

![alt text](figures/math-perspective-correction.png "equation of the perspective correction")

+ compute gradient using Sharr in each spectral band and normalize it
+ detect keypoints on all spectral bands gradient using FAST (for time performance)
+ extract descriptor using ORB (for matche performance)
+ match keypoint of each spectral band to a reference (570 or 710 seem the most valuable -> number of matches)
+ filter matches (distance, position, angle) to remove false positive one (pre-affine transform give epipolar line properties)

![alt text](figures/prespective-feature-matching.jpg "feature matching")

Different type of keypoint extractor has been tested {ORB, AKAZE, BRISK, SURF, FAST} all results can be found in figures/
Two algorithms show great numbers of keypoint after filtering and homography association such as FAST and BRISK.

![alt text](figures/comparaison-keypoint-performances.png "features performances")

Unless BRISK show good number of matched keypoints there computation time is too huge (~79 sec) comparing to FAST (~8 sec) without notable benefits.
SURF is also suitable and allow two parameter nOctaves and nOctaveLayers to increase the number of extracted features, but reducing performances.

The following figure show the Minimum of number of matches between the reference spectra to all others using FAST algorithm.
![alt text](figures/comparaison-keypoint-matching-reference-FAST.png "feature SURF performances")

+ findHomography between detected keypoints
+ perspective correction between each matches (current to reference)
+ estimate reprojection error (rmse+std near to 1 pixel)
+ crop each spectral bands to the minimal bbox

![alt text](figures/prespective-allignement-rmse.jpg "Prespective Reprojection Error")
![alt text](figures/prespective_5.0_false_color.jpg "False Color Corrected Image")

The following figure show the difference between detected point for two bands (red-green)
before (left) and after (right) the perspective correction.

![alt text](figures/perspective-features-matching-scatter.png "Corrected Keypoint")

# Conclusion

You can notive in the above figure that the spatial distribution of the residual angle is equaly distributed.
Our hypothesis is that the nature of the base information (spectral band + different lens) make little difference on the gradient break,
who is detected by the SURF features detector and propagated to the final correction (observed residual).
This is interesting stuff because this equaly distributed residual by angle in the space tend to minimize the resulted correction to his center (gradient).

![alt text](figures/perspective-features-residual.png "Residual Distribution Again Angle")

# Potential related article:

+ https://www.tandfonline.com/doi/abs/10.1080/2150704X.2018.1446564
+ https://citius.usc.es/sites/default/files/publicacions_publicaciones/Alignment%20of%20Hyperspectral%20Images%20Using%20KAZE_Features_v2.pdf
+ https://pdfs.semanticscholar.org/25b6/4d89abdd36e0800da4679813935f055846dd.pdf
+ https://citius.usc.es/sites/default/files/publicacions_publicaciones/Alignment%20of%20Hyperspectral%20Images%20Using%20KAZE_Features_v2.pdf
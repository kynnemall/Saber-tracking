# General points

### DBSCAN to reduce points contributing to saber detection?
* DBSCAN fits fast to a single frame (**less than 10 milliseconds!**) and clusters points well with centroid coordinates and the line segment angle, regardless of distance metric
* Using minimum samples of 2 and epsilon (max distance to belong to a cluster) of 5, the results appear optimal

### How much would DBSCAN slow down the processing if it was done per frame?
* Default compare command includes save statistics and save video (33838 frames in total)
* Processing test video normally takes 12 minutes
* After including DBSCAN fitting to each frame, it takes 14.5 minutes, which is acceptable
* <strong>NOTE: Specifying `n_jobs=-1` takes 15.25 minutes</strong>

<strong>Processing speed</strong>: 14:37 minutes for detection with DBSCAN -> 877 seconds for 33838 frames -> 38.6 FPS

### Minimum length for including line in DBSCAN?
* 40 yields less noise, but misses some key info that 25 picks up
* 30 yields slightly more noise than 40, but this can easily be fixed since these points are near the edge of the frame -> proceed with 30


# Optimizations: round 1
* Removed resize function and included in main function body
* Replacing Parquet with Pandas CSV reduces stats saving time from 0.75 to 0.375 seconds
* Replacing numpy masking with OpenCV functions reduces masking time from 3.1 to 0.993 seconds!
* Replacing Pandas query with `df[df[col] != value]` reduces time from 1.7 to 0.452 seconds!
* Removing the `frame` column as an index (since it's constant) reduces time from 1.7 to 1.2 seconds
* Fitting DBSCAN with `.values` rather than a `DataFrame` reduces time from 1.73 to 1.14 seconds
* Using `sklearnex` sped up DBSCAN in the notebook, but had no effect in the script

### Results
* Base function ran for 17.3986 seconds
* Optimized function ran for 11.5512 seconds
* <strong>33.6% reduction in time to process 500 frames</strong>
* According to `tqdm`, this optimized version ran from script in ~9 seconds! That's ~56 FPS!


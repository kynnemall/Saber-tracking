# General points
<strong>All FPS are reported when processing video, saving both statistics and video with centroids</strong>

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
* Using `sklearnex` sped up DBSCAN in the notebook and in the script by about 400 milliseconds

### Results
* Base function ran for 17.3986 seconds
* Optimized function ran for 11.5512 seconds
* <strong>33.6% reduction in time to process 500 frames</strong>
* According to `tqdm`, this optimized version ran from script in ~9 seconds! That's ~56 FPS!

## Optimizations: round 2
* Using `OpenCV` to mask line data doesn't improve performance over `numpy` masking (~35-36 microseconds)
* Tetsing `Numba` on this function requires the removal of `np.linalg.norm` from the function since they don't currently support the axis argument in this function. 
<strong>This is an open issue on their [Github](https://github.com/numba/numba/pull/7785)</strong>, but it reduces the time to ~8 microseconds even when moving the `norm` function outside the "jitted" function -> 4.5x speedup
However, this will only save 1 second when processing the full 33,838 frame video used for testing, so it won't be incorporated into the main function
* Replacing `Pandas` with pure `Numpy` and saving to HDF5 files instead of CSV allows frames to be processed at a rate of 94 FPS!
* The only way to achieve further performance gains might be to use `FileVideoStream` from `imutils`

### Results
* Without `FileVideoStream` -> 102.4 FPS
* With `FileVideoStream` and `queue_size=250` -> 106.8 FPS
* Is there an optimal `queue_size` to get better performance? 250 seems to be optimal (see README.md in profiles folder)
* Customizing the `FileVideoStream` class to include the masking process when getting frames didn't improve performance

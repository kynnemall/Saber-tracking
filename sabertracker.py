import os
import cv2
import time
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearnex.cluster import DBSCAN

np.random.seed(42)
pd.options.mode.chained_assignment = None  # default='warn'
np.seterr(divide='ignore') # ignore divide by zero when calculating angle

# parameters for Hough Line detection
RHO = 1                 # distance resolution in pixels of the Hough grid
THETA = np.pi / 180     # angular resolution in radians of the Hough grid
THRESHOLD = 20          # minimum number of votes (intersections in Hough grid cell)
MIN_LINE_LENGTH = 5    # minimum number of pixels making up a line
MAX_LINE_GAP = 2       # maximum gap in pixels between connectable line segments
WIDTH = 720             # width to resize the processed video to

def process_video(fname, save_video=False, savename=None, show_video=False, save_stats=False,
                    frame_limit=False):
    if savename == None:
        savename = "saber_tracking.avi"

    if save_video:
        # Initialize video writer to save the results
        out = cv2.VideoWriter(savename, cv2.VideoWriter_fourcc(*'XVID'), 30.0, 
                                 (WIDTH, WIDTH), True)

    cap = cv2.VideoCapture(fname)
    ret, frame = cap.read()
    total_frames = 500 if frame_limit else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)
    frame_num = 0

    output_path = savename.replace(".avi", ".h5")
    # prevent appending to existing file
    if os.path.exists(output_path):
        os.remove(output_path)

    # instantiate DBSCAN for use throughout
    # n_jobs parallelisation introduces too much overhead
    db = DBSCAN(eps=5, min_samples=2)
    data = np.array([])

    while ret:
        ret, frame = cap.read()
        if ret:
            # reset data structure
            data = np.array([])

            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            # these channels were swapped in the notebook
            b = cv2.inRange(frame[:, :, 2], 200, 255)
            r = cv2.inRange(frame[:, :, 0], 180, 255)

            # convert to HSV for more masking options
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = cv2.inRange(hsv[:, :, 2], 170, 255)
            s = cv2.inRange(hsv[:, :, 1], 140, 175)

            # combine masks into one
            m1 = cv2.bitwise_and(b, s)
            m2 = cv2.bitwise_and(r, v)
            mask = cv2.bitwise_or(m1, m2)

            # Run Hough on masked image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(mask, RHO, THETA, THRESHOLD, np.array([]), MIN_LINE_LENGTH, MAX_LINE_GAP)

            # process lines
            if isinstance(lines, np.ndarray):
                lines = lines[:, 0, :]
                cx = lines[:, [0, 2]].mean(axis=1).reshape(-1, 1)
                cy = lines[:, [1, 3]].mean(axis=1).reshape(-1, 1)
                frames = np.zeros(cx.shape) + frame_num
                slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])
                angles = np.rad2deg(np.arctan(slopes)).reshape(-1, 1)
                lengths = np.linalg.norm(lines[:, :2] - lines[:, 2:], axis=1).reshape(-1, 1)
                data = np.concatenate((frames, cx, cy, angles, lengths), axis=1)

                # filter by edge conditions and line length
                xedge_mask = np.logical_and(data[:, 1] > 100, data[:, 1] < 540)
                yedge_mask = np.logical_and(data[:, 2] > 50, data[:, 2] < 310)
                len_mask = np.logical_and(data[:, -1] > 10, data[:, -1] < 50)
                mask = np.logical_and(len_mask, xedge_mask)
                mask = np.logical_and(mask, yedge_mask)
                data = data[len_mask]

            # perform clustering to reduce data
            if data.size > 0:
                db.fit(data[:, 1:4])
                data = np.concatenate((data, db.labels_.reshape(-1, 1)), axis=1)
                data = data[data[:, -1] != -1]
                if data.size > 0:
                    for i in np.unique(data[:, -1]):
                        centroid = data[data[:, -1] == i][:, 1:3].mean(axis=0).astype(int)
                        cv2.drawMarker(frame, centroid, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

            if save_stats and data.size > 0:
                # save tracking data
                if not os.path.exists(output_path):
                    with h5py.File(output_path, "w") as hf:
                        hf.create_dataset("data", data=data, 
                                          compression="gzip",
                                          chunks=True, maxshape=(None, 6))
                else:
                    with h5py.File(output_path, "a") as hf:
                        # append new data
                        new_shape = (hf["data"].shape[0] + data.shape[0])
                        hf["data"].resize(new_shape, axis=0)
                        hf["data"][-data.shape[0]:] = data

            resized = cv2.resize(frame, (WIDTH, WIDTH))

            if show_video:
                cv2.imshow("Frame", frame)
            if save_video:
                out.write(resized)
            frame_num += 1
            pbar.update(1)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                cv2.waitKey(-1) # wait until any key is pressed
            if frame_limit and frame_num == 500:
                break

    cap.release()
    if save_video:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
    return total_frames

if __name__ == "__main__":
    # set up argparser for CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("-sv", "--vidsave", action="store_true", default=False, help="save video after processing")
    parser.add_argument("-sh", "--show", action="store_true", default=False, help="show video while processing")
    parser.add_argument("-ss", "--tracksave", action="store_true", default=False, help="save tracking data")
    parser.add_argument("-p", "--trackproc", action="store_true", default=False, help="process tracking data")
    parser.add_argument("-f", "--filepath", default="test_video.mp4", help="path to file")
    parser.add_argument("-sn", "--savename", default=None, help="name for saving processed video file")
    parser.add_argument("-l", "--limit", default=False, action="store_true", help="limit processing to the first 500 frames")
    args = parser.parse_args()

    if args.trackproc:
        # process tracking data
        pass
    else:
        start = time.time()
        # process video
        frames = process_video(args.filepath, args.vidsave, args.savename, args.show, args.tracksave,
                        args.limit)
        end = time.time()
        diff = end - start
        fps = frames / diff
        print(f"Took {diff:.2f} seconds to process {frames} frames")
        print(f"Processing speed -> {fps:.1f} FPS")

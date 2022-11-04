# script from: https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
import cv2
import numpy as np
from skimage.morphology import skeletonize

im_w = 720

def resize(img):
    return cv2.resize(img, (im_w, im_w))

cap = cv2.VideoCapture("test_video.mp4")
ret, frame = cap.read()
frame_num = 0

# Initialize video writer to save the results
out = cv2.VideoWriter('hough_lines_tracking.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 30.0, 
                         (im_w, im_w), True)

# upper and lower bounds for thresholding
sensitivity = 60
l_b = np.array([0, 0, 255 - sensitivity])   # lower hsv bound for white
u_b = np.array([255, sensitivity, 255])     # upper hsv bound for white

# parameters for Hough Line detection
rho = 1                 # distance resolution in pixels of the Hough grid
theta = np.pi / 180     # angular resolution in radians of the Hough grid
threshold = 40          # minimum number of votes (intersections in Hough grid cell)
min_line_length = 25    # minimum number of pixels making up a line
max_line_gap = 5        # maximum gap in pixels between connectable line segments

while ret:
    ret, frame = cap.read()

    # convert to HSV for simplicity
    blur = cv2.GaussianBlur(frame, (3, 3), 0) # (5,5) is better when there's no sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    blur = cv2.filter2D(blur, ddepth=-1, kernel=kernel)
    m1 = blur[:, :, 1] > 200
    m2 = blur[:, :, 2] > 200
    m3 = blur[:, :, 0] > 240
    mask = (np.logical_and(m1, m2) + m3).astype(np.uint8)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        mask, rho, theta, threshold, np.array([]),
        min_line_length, max_line_gap
    )

    gray = np.zeros(frame.shape[:2], np.uint8)
    if isinstance(lines, np.ndarray):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(gray, (x1, y1), (x2, y2), 255, 2)
                x_diff = x1 - x2
                y_diff = y1 - y2
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                degrees = np.rad2deg(np.arctan(y_diff / x_diff))
                cv2.drawMarker(frame, centroid, (0, 255, 0),
                               markerType=cv2.MARKER_CROSS, thickness=2)

#    # Drawing contours is only good for visualising saber detection
#    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    for i, contour in enumerate(contours):
#        if 800 > cv2.contourArea(contour) > 60:
#            cv2.drawContours(frame, contours, i, 255, -1)
    resized = resize(frame)
    
    cv2.imshow("Frame", resized)
    print(f"Frame {frame_num}")
    out.write(resized)
    frame_num += 1

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) # wait until any key is pressed

cap.release()
out.release()
cv2.destroyAllWindows()

# script from: https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
import cv2
import numpy as np
from skimage.morphology import skeletonize

im_w = 720

def resize(img):
    return cv2.resize(img, (im_w, im_w))

cap = cv2.VideoCapture("Ludosport_birthday_sparring.mp4")
ret, frame = cap.read()

# Initialize video writer to save the results
out = cv2.VideoWriter('hough_lines_tracking.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 30.0, 
                         (im_w, im_w), True)

# upper and lower bounds for thresholding
sensitivity = 60
l_b = np.array([0, 0, 255 - sensitivity])   # lower hsv bound for white
u_b = np.array([255, sensitivity, 255])     # upper hsv bound for white

# parameters for Hough Line detection
rho = 1                         # distance resolution in pixels of the Hough grid
theta = np.pi / 180             # angular resolution in radians of the Hough grid
threshold = 30                  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 80            # minimum number of pixels making up a line
max_line_gap = 25               # maximum gap in pixels between connectable line segments

while ret:
    ret, frame = cap.read()

    # convert to HSV for simplicity
    # blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, l_b , u_b)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1) # dilate mask to make it easier

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(mask, low_threshold, high_threshold)
    # edges = skeletonize(mask > 100).astype(np.uint8)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]),
        min_line_length, max_line_gap
    )

    if isinstance(lines, np.ndarray):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), 255, 2)
    print(f"Found {lines.size} lines")
    resized = resize(frame)
    
    cv2.imshow("frame", resized)
    out.write(resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) #wait until any key is pressed

cap.release()
out.release()
cv2.destroyAllWindows()

# script from: https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
import cv2
import numpy as np

im_w = 512

def resize(img):
    return cv2.resize(img, (im_w, im_w))

cap = cv2.VideoCapture("Ludosport_birthday_sparring.mp4")
ret, frame = cap.read()

# Initialize video writer to save the results
out = cv2.VideoWriter('hough_lines_tracking.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 30.0, 
                         (im_w, im_w), True)

# l_b=np.array([0,230,170]) # lower hsv bound for red
# u_b=np.array([255,255,220]) # upper hsv bound to red

sensitivity = 50
l_b = np.array([0, 0, 255 - sensitivity]) # lower hsv bound for white
u_b = np.array([255, sensitivity, 255]) # upper hsv bound for white

while ret:
    ret, frame = cap.read()

    # convert to HSV for simplicity
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, l_b , u_b)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(mask, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    if isinstance(lines, np.ndarray):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
    resized = resize(frame)
    
    cv2.imshow("frame", resized)
    out.write(resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

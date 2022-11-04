# script from: https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
import cv2
import numpy as np

im_w = 512

def resize(img):
    return cv2.resize(img, (im_w, im_w))

cap = cv2.VideoCapture("test_video.mp4")
ret, frame = cap.read()

# Initialize video writer to save the results
out = cv2.VideoWriter('intensity_based_tracking.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 30.0, 
                         (im_w, im_w), True)

# l_b=np.array([0,230,170]) # lower hsv bound for red
# u_b=np.array([255,255,220]) # upper hsv bound to red

sensitivity = 75
l_b = np.array([0, 0, 255 - sensitivity]) # lower hsv bound for white
u_b = np.array([255, sensitivity, 255]) # upper hsv bound for white

while ret:
    ret, frame = cap.read()

    # convert to HSV for simplicity
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, l_b , u_b)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 5:
            # fit ellipse to contour
            ellipse = cv2.fitEllipse(contour)
            w = ellipse[1][0]
            h = ellipse[1][1]
            if w > h and h > 0:
                r = w / h
            elif h > w and w > 0:
                r = h / w
            else:
                r = 0

            # if major to minor ratio is in a certain range, plot the ellipse
            if 50 > r > 5:
                cv2.ellipse(frame, ellipse, (0, 255, 0), 1, cv2.LINE_AA)

    resized = resize(frame)
    
    cv2.imshow("frame", resized)
    out.write(resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

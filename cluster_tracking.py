import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

im_w = 512

def resize(img):
    return cv2.resize(img, (im_w, im_w))

cap = cv2.VideoCapture("test_video.mp4")
ret, frame = cap.read()

# Initialize video writer to save the results
out = cv2.VideoWriter('intensity_based_tracking.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'), 30.0, 
                         (im_w, im_w), True)
fit_kmeans = True

while ret:
    ret, img = cap.read()

    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    if fit_kmeans:
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=0,
                    batch_size=4096).fit(image_2D)
        clustered = kmeans.cluster_centers_[kmeans.labels_]
    else:
        clustered = kmeans.predict(image_2D)

    clustered_3D = clustered.reshape(img.shape)
    mask = cv2.cvtColor(clustered_3D.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    mask = (mask > 150).astype(np.uint8)

    # settings for Hough Lines P
    rho = 1                         # distance resolution in pixels of the Hough grid
    theta = np.pi / 180             # angular resolution in radians of the Hough grid
    threshold = 40                  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40            # minimum number of pixels making up a line
    max_line_gap = 10               # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(mask, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    gray = np.zeros(img.shape[:2], np.uint8)
    if isinstance(lines, np.ndarray):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(gray, (x1, y1), (x2, y2), 255, 2)
                length = abs(x1 - x2) + abs(y1 - y2)
                if 100 > length > 40:
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    cv2.drawMarker(img, centroid, (0, 255, 0),
                                   markerType=cv2.MARKER_CROSS, thickness=2)

    resized = resize(img)
    
    cv2.imshow("frame", resized)
    out.write(resized)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) # wait until any key is pressed

cap.release()
out.release()
cv2.destroyAllWindows()

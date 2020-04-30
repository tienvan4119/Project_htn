import numpy as np
import cv2 as cv

img = cv.imread('phone2phone.JPG')
sift = cv.xfeatures2d.SIFT_create()

kp = sift.detect(img, None)
img = cv.drawKeypoints(img, kp, img)
cv.imwrite('sift_keypoints.jpg', img)
img = cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints.jpg', img)
cv.imshow("test", img)
cv.waitKey(0)
cv.destroyAllWindows()

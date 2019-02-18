import cv2
import numpy as np

im = cv2.imread('processing.jpg',cv2.IMREAD_GRAYSCALE)
im2 = im.copy()
mask = np.zeros((np.array(im.shape)+2), np.uint8)
cv2.floodFill(im, mask, (0,0), (255))
im = cv2.erode(im, np.ones((3,3)))
im = cv2.bitwise_not(im)
im = cv2.bitwise_and(im,im2)
# cv2.imshow('show', im)
cv2.imwrite('fin.png',im)
# cv2.waitKey()
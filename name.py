import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('processing.jpg',0)

#display image in window
#cv2.imshow('image',img) #@param - windowname, image to be displayed

horizontal_inv = cv2.bitwise_not(img)
#perform bitwise_and to mask the lines with provided mask
masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
masked_img = cv2.bitwise_and(masked_img, masked_img, mask=horizontal_inv)
# masked_img = cv2.bitwise_or(masked_img, masked_img, mask=horizontal_inv)
# masked_img = cv2.bitwise_or(masked_img, masked_img, mask=horizontal_inv)
# masked_img = cv2.bitwise_or(masked_img, masked_img, mask=horizontal_inv)
# masked_img = cv2.bitwise_and(masked_img, masked_img, mask=horizontal_inv)
#reverse the image back to normal
masked_img_inv = cv2.bitwise_not(masked_img)

kernel = np.ones((10,10),np.uint8)
dilation = cv2.dilate(masked_img_inv,kernel,iterations = 7) # to remove blackline noise
cv2.imwrite("result1.jpg", dilation)
# ret,thresh2 = cv2.threshold(dilation,254,255,cv2.THRESH_BINARY_INV) 
# thresh2=cv2.bitwise_not(thresh2)
# # cv2.imshow("masked img", masked_img_inv)
# cv2.imwrite("result2.jpg", thresh2)
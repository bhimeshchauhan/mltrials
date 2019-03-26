# import cv2
import numpy as np
#
# img = cv2.imread('img/test2.jpg')
# blur = cv2.GaussianBlur(img, (15, 15), 2)
# hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
# lower_green = np.array([0, 255, 255])
# upper_green = np.array([153, 255, 255])
# mask = cv2.inRange(hsv, lower_green, upper_green)
# masked_img = cv2.bitwise_and(img, img, mask=mask)
# cv2.imwrite('color.png', masked_img)


#
# img = cv2.imread('img/test1.jpg')
# sensitivity = 50
#
#
# blur = cv2.GaussianBlur(img, (15, 15), 2)
# lower_green = np.array([34, 139, 34])
# upper_green = np.array([73, 121, 107])
# mask = cv2.inRange(blur, lower_green, upper_green)
# masked_img = cv2.bitwise_and(img, img, mask=mask)
# cv2.imwrite('color.png', masked_img)


# ##############################

# img = cv2.imread('img/test1.jpg')
# blur = cv2.GaussianBlur(img, (15, 15), 2)
# hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
# lower_green = np.array([37, 0, 0])
# upper_green = np.array([100, 189, 182])
# mask = cv2.inRange(hsv, lower_green, upper_green)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# masked_img = cv2.bitwise_and(img, img, mask=opened_mask)
# cv2.imwrite('color.png', masked_img)


# ###############################
import cv2
import numpy as np

image = cv2.imread('img/test9.jpg')
image = cv2.blur(image,(1,1))

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsv.jpg', hsv)

# define range of green/red color in HSV
lower_green = np.array([0,70,100])
upper_green = np.array([50,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_green, upper_green)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(image,image, mask= mask)

cv2.imwrite('res.jpg', res)
# cv2.imshow('image',image)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
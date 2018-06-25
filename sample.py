import cv2
import numpy as np


img = cv2.imread('original.jpg')
img_D = img
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


H = hsv[:,:,0]
S = hsv[:,:,1]
V = hsv[:,:,2]
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(17,17))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))


# processing on H for extraction of pink parts

H = cv2.dilate(H,kernel1,iterations = 10)
ret,th1 = cv2.threshold(H,127,255,cv2.THRESH_BINARY)
th1 = cv2.erode(th1,kernel1,iterations = 10)
th1  = cv2.morphologyEx(th1 , cv2.MORPH_OPEN, kernel1)

_, contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

(x,y),radius = cv2.minEnclosingCircle(contours[0])
center = (int(x),int(y))
radius = int(radius)
img1 = cv2.circle(img,center,radius,(0,255,0),7)

cv2.imwrite("sample1.jpg", img1)

img1 = cv2.resize(img1, (960, 540))                   
cv2.imshow("output", img1)                           
cv2.waitKey(0) 
cv2.destroyAllWindows()


# processing on V for extraction of cross

V = cv2.erode(V,kernel3,iterations = 5)
th3 = cv2.morphologyEx(V, cv2.MORPH_OPEN, kernel3)
ret,th3 = cv2.threshold(th3,127,255,cv2.THRESH_BINARY)
th3 = cv2.dilate(th3,kernel2,iterations = 15)



_, contours2, hierarchy = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

x,y,w,h = cv2.boundingRect(contours2[0])
img2 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),7)

cv2.imwrite("sample2.jpg", img2)

img2 = cv2.resize(img2, (960, 540))                   
cv2.imshow("output", img2)                           
cv2.waitKey(0) 
cv2.destroyAllWindows()





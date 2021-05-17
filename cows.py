import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
from skimage.metrics import structural_similarity
import time

img = cv2.imread('DJI_0014b.jpg', 1)
img_gs = cv2.imread('DJI_0014b.jpg', 0)

scale_percent = 45 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

blur = cv2.bilateralFilter(img, 9, 75, 75)

resized = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
#291:304, 119:124
# black 205:218, 162:168
crop_img = resized[400:420, 560:570]
cv2.imshow('image',crop_img)

# ix,iy = -1,-1
# # # # # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     global ix,iy
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#         ix,iy = x,y
# #
# # # # # Create a black image, a window and bind the function to window
# # # # # img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
# # # #
# while(1):
#    cv2.imshow('image',resized)
#    k = cv2.waitKey(20) & 0xFF
#    if k == 27:
#        break
#    elif k == ord('a'):
#        print (ix,iy)
# cv2.destroyAllWindows()

hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

#hsv[:,:,0] += 120
# for brown cows
# ret,thresh = cv2.threshold(hsv[:,:,0],136,255,cv2.THRESH_TOZERO_INV)
# ret,thresh1 = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY)
#
# ret,thresh2 = cv2.threshold(hsv[:,:,1],255,255,cv2.THRESH_TOZERO_INV)
# ret,thresh3 = cv2.threshold(thresh2,80,255,cv2.THRESH_BINARY)

#dst = cv2.bitwise_and(thresh1,thresh3)

# ret,thresh4 = cv2.threshold(dst,105,255,cv2.THRESH_TOZERO_INV)

### For black cows
ret,thresh = cv2.threshold(hsv[:,:,0],120,255,cv2.THRESH_TOZERO_INV)
ret,thresh1 = cv2.threshold(thresh,70,255,cv2.THRESH_BINARY)
#


ret,thresh2 = cv2.threshold(hsv[:,:,1],35,255,cv2.THRESH_BINARY_INV)

ret,thresh4 = cv2.threshold(hsv[:,:,2],80,255,cv2.THRESH_TOZERO_INV)
ret,thresh3 = cv2.threshold(thresh4,50,255,cv2.THRESH_BINARY)

dst0 = cv2.bitwise_and(thresh1,thresh2)
dst1 = cv2.bitwise_and(dst0, thresh3)

hist = cv2.calcHist([thresh1], [0], None, [255], [1, 255])

plt.plot(hist, color='b')

hist = cv2.calcHist([thresh2], [0], None, [255], [1, 255])

plt.plot(hist, color='g')

hist = cv2.calcHist([thresh3], [0], None, [255], [1, 255])

plt.plot(hist, color='r')


#hist = cv2.calcHist([hsv], [0,1], None, [180, 256], [0, 180, 0, 256])

plt.show()

cv2.namedWindow('Test')        # Create a named window
cv2.moveWindow('Test', 40,30)  # Move it to (40,30)
cv2.imshow('thresh1', thresh2)
cv2.imshow('thresh', thresh3)
cv2.imshow('equ', dst1)
cv2.imshow('equ2', thresh1)
cv2.waitKey()
cv2.destroyAllWindows()

#cap = cv2.VideoCapture('videos/DJI_0026.mov')

# # if cap.isOpened() == False:
# #         print('Cannot open video stream')
# #
# # ret, frame = cap.read()
# height, width = img.shape[:2]
print(height, width)
# crop_default = img[0:int(height), 0:int(width/2)]
#
# cv2.waitKey()
# #
# # gray_default = cv2.cvtColor(crop_default, cv2.COLOR_BGR2GRAY)
#
# fgbg = cv2.createBackgroundSubtractorMOG2(
#     history=10,
#     varThreshold=8,
#     detectShadows=False)
#
# fgmask = fgbg.apply(crop_default)
#
# while True:
#     ret, frame = cap.read()
#
#     #frame[:,:,1] = np.zeros([frame.shape[0], frame.shape[1]])
#
#     crop = frame[0:int(height), 0:int(width/2)]
#
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     light = hsv[:,:,2]
#     h_edges0 = cv2.Canny(hsv[:,:,2], 5, 200)
#     h_edges2 = cv2.Canny(hsv[:,:,2], 50, 300)
#     h_edges3 = cv2.Canny(hsv[:,:,2], 100, 400)
#
#     #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     #gray = cv2.cvtColor(hsv[:,:,2], cv2.COLOR_BGR2GRAY)
#
#     res,thresh3 = cv2.threshold(light,180,255,cv2.THRESH_TRUNC)
#
#     images3 = np.hstack((thresh3, light, hsv[:,:,2]))
#
#     #gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
#     #equ = cv2.equalizeHist(gray)
#     #blur = cv2.GaussianBlur(gray, (3,3), 0)
#
#
#     #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 205, 1)
#
#     #res, thresh = cv2.threshold(blur, 217, 255, cv2.THRESH_BINARY)
#
#     #edges = cv2.Canny(gray, 10, 200)
#
#     # Extract the foreground
#     edges_foreground = cv2.bilateralFilter(thresh3, 9, 75, 75)
#     foreground = fgbg.apply(edges_foreground)
#
#     # Smooth out to get the moving area
#     kernel = np.ones((50,50),np.uint8)
#     foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
#
#     # Applying static edge extraction
#     edges_foreground = cv2.bilateralFilter(thresh3, 9, 25, 175)
#     edges_filtered = cv2.Canny(edges_foreground, 60, 120)
#
#     # Crop off the edges out of the moving area
#     cropped = (foreground // 255) * edges_filtered
#
#     # Stacking the images to print them together for comparison
#
#     images = np.hstack((light, edges_filtered, cropped))
#     images2 = np.hstack((edges_foreground, foreground, cropped))
#
#     contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     img2 = frame.copy()
#     index = -1
#     thickness = 1
#     color = (255,0,255)
#
#     cv2.drawContours(img2, contours, index, color, thickness)
#
#     if ret == True:
#
#     #    histrB = cv2.calcHist([gray_filtered],[0],None,[256],[0,256])
#     #    plt.plot(histrB)
#
#         #cv2.imshow('Stacked', images)
#         cv2.imshow('Stacked2', img2)
#         #cv2.imshow('Stacked3', images3)
#
#         #good
#         #cv2.imshow('Canny', edges)
#         #good
#         #cv2.imshow('Aftermath', thresh)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

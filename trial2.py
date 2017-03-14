import cv2
import numpy as np
cap = cv2.VideoCapture('cellvid.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


while(1):
    # Take each frame
    _, frame = cap.read()
    
    #blurring 
    blur = cv2.bilateralFilter(frame,9,75,75)


    #thresholding
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([140,30,75])
    upper_purple = np.array([165,180,255])
    # Threshold the HSV image to get only purple colors
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(hsv,hsv, mask= mask)


    opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations = 2)

    edges = cv2.Canny(opening[:,:,2],100,200)

###########
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = edges.copy()
    cv2.drawContours(img, contours, -1, (0,255,0), 2)

    cv2.namedWindow('contours',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('contours', 600,600)
  ########### 
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 600,600)
    
    cv2.namedWindow('final',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('final', 600,600)

    cv2.namedWindow('edges',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('edges', 600,600)

    cv2.imshow('frame',frame)
    cv2.imshow('final',opening)
    cv2.imshow('edges',edges)
    ################
    cv2.imshow('contours',img)

    k = cv2.waitKey(1500) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
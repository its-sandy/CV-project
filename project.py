import cv2
import numpy as np
cap = cv2.VideoCapture('cellvid.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1024,768))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


while(cap.isOpened()):
    
    ret, frame = cap.read()
    if ret==True:
    
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

        rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 600,600)
        
        cv2.namedWindow('final',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('final', 600,600)

        cv2.namedWindow('edges',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('edges', 600,600)

        cv2.imshow('frame',frame)
        cv2.imshow('final',opening)
        cv2.imshow('edges',edges)

        out.write(rgb)
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


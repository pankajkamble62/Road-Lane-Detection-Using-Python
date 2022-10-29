
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import cv2
import imutils
import time


cap = cv2.VideoCapture('testvideo2.mp4')


while(cap.isOpened()):

   
    ret, frame = cap.read()
    
    snip = frame[500:700,300:900]
    cv2.imshow("Snip",snip)

    # creating polygon 
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
    pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    cv2.imshow("Mask", mask)

    
    masked = cv2.bitwise_and(snip, snip, mask=mask)
    cv2.imshow("Region of Interest", masked)

    # converting to grayscale     
    frame = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    thresh = 200
    frame = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Black/White", frame)

    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    edged = cv2.Canny(blurred, 30, 1-+50)
    cv2.imshow("Edged", edged)

    lines = cv2.HoughLines(edged, 1, np.pi / 180, 25)

    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []

    
    if lines is not None:

       
        for i in range(0, len(lines)):

            
            for rho, theta in lines[i]:

                
                if theta < np.pi/2 and theta > np.pi/4:
                    rho_left.append(rho)
                    theta_left.append(theta)

                if theta > np.pi/2 and theta < 3*np.pi/4:
                    rho_right.append(rho)
                    theta_right.append(theta)

                   
    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)

    if left_theta > np.pi/4:
        a = np.cos(left_theta); b = np.sin(left_theta)
        x0 = a * left_rho; y0 = b * left_rho
        offset1 = 250; offset2 = 800
        x1 = int(x0 - offset1 * (-b)); y1 = int(y0 - offset1 * (a))
        x2 = int(x0 + offset2 * (-b)); y2 = int(y0 + offset2 * (a))

        cv2.line(snip, (x1, y1), (x2, y2), (0, 255, 0), 6)

    if right_theta > np.pi/4:
        a = np.cos(right_theta); b = np.sin(right_theta)
        x0 = a * right_rho; y0 = b * right_rho
        offset1 = 290; offset2 = 800
        x3 = int(x0 - offset1 * (-b)); y3 = int(y0 - offset1 * (a))
        x4 = int(x0 - offset2 * (-b)); y4 = int(y0 - offset2 * (a))

        cv2.line(snip, (x3, y3), (x4, y4), (255, 0, 0), 6)


    if left_theta > np.pi/4 and right_theta > np.pi/4:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

        overlay = snip.copy()
        
        cv2.fillConvexPoly(overlay, pts, (0, 255, 0))

        opacity = 0.4
        cv2.addWeighted(overlay, opacity, snip, 1 - opacity, 0, snip)

    cv2.imshow("Lined", snip)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import multiprocessing
import cv2
import helper

cap = cv2.VideoCapture(1)
cv2.CAP_PROP_FRAME_HEIGHT
capWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
capHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
capArea = capWidth * capHeight


print multiprocessing.cpu_count()

while(True):
    ret, img_raw = cap.read()
    if not ret:
        print "no frame data"
        exit(1)

    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # edge detection and find contours
    img_canny = cv2.Canny(img_gray, 100, 200)
    _, contours, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ignore contours that are too small:
    # define too small: the max area of the contour is smaller than 10% of the screen area
    contours = helper.filterSmallContours(contours, capArea, 0.1)

    # ignore contours looks no where like a quadrilateral
    contours = helper.filterNonQuadrilateral(contours, 0.1)

    if contours:
        if len(contours)> 1:
            print "multiple"
        for contour in contours:
            contour = helper.findVerticesCW(contour)

            # get binary image
            thresh, img_bw = cv2.threshold(img_gray, 127, 255,
                                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            ortho_bw, trans = helper.getOrthophoto(
                img_bw, contour, helper._sudokuCtrPts2D)
            ortho_raw = cv2.warpPerspective(img_raw, trans, (400,400))
            
            _, cnts, _= cv2.findContours(ortho_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnts = helper.filterNonQuadrilateral(cnts, 0.02)

            cv2.drawContours(ortho_raw, cnts, -1, (0,150,0), 2)
            for i in range(len(contour)):
                pos = contour[i]
                cv2.putText(img_raw, str(i), map(tuple, pos)[
                    0], cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

        cv2.drawContours(img_raw, contours, -1, (0, 150, 0), 2)
        cv2.imshow('ortho', ortho_raw)
    cv2.imshow('frame', img_raw)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

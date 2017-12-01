import cv2
import numpy as np
import helper

img_raw = cv2.imread('sudoku.jpg')

img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
# edge detection and find contours
img_canny = cv2.Canny(img_gray, 100, 200)
_, contours, hierarchy = cv2.findContours(
    img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print contours
# ignore contours that are too small:
# define too small: the max area of the contour is smaller than 20% of the screen area
contours = helper.filterSmallContours(contours, 640*480, thresh=0.2)

# ignore contours looks no where like a quadrilateral
# contours = helper.filterNonQuadrilateral(contours, esp=0.05)
contours = [helper.simplifyContour(contours[0], 0.02)]
cv2.drawContours(img_raw, contours,-1, (255,0,0),4)
if contours:
    if len(contours) > 1:
        print "multiple"
    for contour in contours:
        contour = helper.findVerticesCW(contour)
        # get binary image
        thresh, img_bw = cv2.threshold(img_gray, 127, 255,
                                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ortho_bw, trans = helper.getOrthophoto(
            img_bw, contour, helper.__sudoku_ctrl_pts2D)
        ortho_raw = cv2.warpPerspective(img_raw, trans, (400, 400))
        _, cnts, _ = cv2.findContours(
            ortho_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = helper.filterNonQuadrilateral(cnts, 0.02)
        cv2.drawContours(ortho_raw, cnts, -1, (0, 150, 0), 2)
        for i in range(len(contour)):
            pos = contour[i]
            cv2.putText(img_raw, str(i), map(tuple, pos)[
                0], cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.drawContours(img_raw, contours, -1, (0, 150, 0), 2)
    cv2.imshow('ortho', ortho_raw)
cv2.imshow('frame', img_raw)
cv2.waitKey(0)
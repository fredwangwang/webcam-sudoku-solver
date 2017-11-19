import numpy as np
import cv2
import json

import helper

pic = 'sudoku2.jpg'

img_raw = cv2.imread(pic, cv2.IMREAD_COLOR)

# get binary image
img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
thresh, img_bw = cv2.threshold(img_gray, 127, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# img_bw = cv2.adaptiveThreshold(img_gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 13, 15)

# find contours w/ edge detection
img_canny = cv2.Canny(img_gray, 100, 200)
_, contours, hierarchy = cv2.findContours(
    img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# simplify the contour and order the vertices in clockwise order
isQuad, contour = helper.isQuadrilateral(contours[0])
if not isQuad:
    print "This is not a quadrilateral"
contour = helper.findVerticesCW(contour)


cv2.drawContours(img_raw, [contour], -1, (0, 255, 0), 2)


_sudoku_control_points = np.array([[[000, 000, 0]],
                                   [[100, 000, 0]],
                                   [[100, 100, 0]],
                                   [[000, 100, 0]]], dtype="double")

camera_matrix = np.array(
    [[531, 0, 320],
     [0, 531, 240],
     [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

image_points = np.array([
    [(359, 391)],     # Nose tip
    [(399, 561)],     # Chin
    [(337, 297)],     # Left eye left corner
    [(513, 301)]     # Right eye right corne
], dtype="double")

# 3D model points.
model_points = np.array([
    [(0.0, 0.0, 0.0)],             # Nose tip
    [(0.0, -330.0, -65.0)],        # Chin
    [(-225.0, 170.0, -135.0)],     # Left eye left corner
    [(225.0, 170.0, -135.0)]
], dtype="double")

print np.shape(_sudoku_control_points)
print np.shape(model_points)
print np.shape(contour)
print np.shape(image_points)

success, rvec, tvec = cv2.solvePnP(
    _sudoku_control_points, contour.astype(float), camera_matrix, dist_coeffs)
print helper.findVerticesCW(contours[0])

# run hough line transform
lines = cv2.HoughLines(img_bw, 1, 1.5 * np.pi / 180, 270)

# draw lines
for line in lines:
    rho = line[0][0]
    theta = line[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img_raw, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('bin', img_bw)
cv2.imshow('canny', img_canny)
cv2.imshow('raw', img_raw)
cv2.waitKey(0)
cv2.destroyAllWindows()

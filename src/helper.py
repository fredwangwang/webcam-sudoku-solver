'''Helper functions for sudoku solver project'''

import numpy as np
import cv2
import json

_sudoku_control_points = np.array([[[000, 000, 0]],
                                   [[100, 000, 0]],
                                   [[100, 100, 0]],
                                   [[000, 100, 0]]])


def simplifyContour(contour, eps=0.1):
    '''Given a numpy array of points (contour), return the simplifed contour

    simplifyContour(contour, eps) -> contour'''
    epsilon = eps * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def isQuadrilateral(contour, eps=0.1):
    '''Given a numpy array of points (contour), return whether the contour is a quadrilateral,
    and the simplified contour

    isQuadrilateral(contour, eps) -> isQuad, contour'''
    if len(contour) == 4:
        return True, contour
    contour = simplifyContour(contour, eps)
    if len(contour) == 4:
        return True, contour
    return False, contour


def findVerticesCW(contour):
    '''Given a numpy array of points (contour), return a numpy array of four extreme vertices of the shape Nx1(2channel).
    The order of the returned data would be clockwise, starting from top-left corner. For performance reason,
    this function should only be called with a contour that has few control points

    findVerticesCW(contour) -> verticesCW
    '''
    '''
    structure:
    [                           # np.ndarray N 
        [                       # np.ndarray point
            [                   # np.ndarray 2channel
                129,            # x channel 1
                70              # y channel 2
            ]
        ]
    ]
    '''
    min_x = min_y = 999999
    max_x = max_y = 0
    for pt in contour:
        try:
            assert len(pt) == 1 and len(pt[0]) == 2
        except AssertionError:
            print 'The given data is not a valid contour. Nx1 2channel data expected.'
            exit(1)
        pt = pt[0]
        x = pt[0]
        y = pt[1]
        max_x = x if max_x < x else max_x
        max_y = y if max_y < y else max_y
        min_x = x if min_x > x else min_x
        min_y = y if min_y > y else min_y

    def distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    tl = np.array([min_x, min_y])
    tr = np.array([max_x, min_y])
    bl = np.array([min_x, max_y])
    br = np.array([max_x, max_y])
    bbox = (tl, tr, br, bl)

    result = np.ndarray(shape=(4, 1, 2), dtype=np.ndarray)
    for i in range(len(bbox)):
        ctr_pt = bbox[i]
        min_dist = 9999999
        for pt in contour:
            dist = distance(ctr_pt, pt[0])
            if min_dist > dist:
                result[i] = pt
                min_dist = dist
    return result
'''
// Check the "circularity" ratio of the outer region, which is
		// the ratio of area to perimeter squared: R = 4*pi*A/P^2.
		// R is 1 for a circle, and pi/4 for a square.
		double P1 = arcLength(contours[i1], true);
		double A1 = contourArea(contours[i1]);
		if (4 * 3.1415 * A1 / (P1 * P1) < 3.1415 / 4)
			// Let's say that we want our region to be at least as round as a square.
			continue;'''
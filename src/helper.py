'''Helper functions for sudoku solver project'''

import numpy as np
import cv2

_sudokuCtrPts3D = np.array([[[000, 000, 0]],
                            [[400, 000, 0]],
                            [[400, 400, 0]],
                            [[000, 400, 0]]])

_sudokuCtrPts2D = np.array([[[0, 0]],
                            [[400, 0]],
                            [[400, 400]],
                            [[0, 400]]])


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

    # TODO: refactor required.
    # The contour returned is guareeteed to be CCW, take advantage of that

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

    result = np.ndarray(shape=(4, 1, 2), dtype="int")
    for i in range(len(bbox)):
        ctr_pt = bbox[i]
        min_dist = 9999999
        for pt in contour:
            dist = distance(ctr_pt, pt[0])
            if min_dist > dist:
                result[i] = pt
                min_dist = dist
    return result

def getOrthophoto(img, bbox, ctrPts, trans="projective"):
    '''Given the image and coordinate pairs, return the orthophoto and transform

    getOrthophoto(img, bbox, ctrPts, trans) -> orthophoto, trans
    '''
    trans = cv2.getPerspectiveTransform(bbox.astype(np.float32), ctrPts.astype(np.float32))
    return cv2.warpPerspective(img, trans, (400,400)), trans
    # affine = cv2.getAffineTransform(bbox[1:].astype(np.float32), ctrPts[1:].astype(np.float32))
    # return cv2.warpAffine(img, affine, (400,400))


def filterSmallContours(contours, area, thresh=0.1):
    '''Given a list of contours, filter out the contours that is smaller than
    area*thresh. Return the qulified contours

    filterSmallContours(contours, area, thresh) -> contours
    '''
    return [x for x in contours if cv2.contourArea(x) > area * thresh]


def filterNonQuadrilateral(contours, esp=0.1):
    '''Given a list of contours, filter out the contours that is no like a 
    quadrilateral shape. The returned contours are simplified.

    filterNonQuadrilateral(contours, esp) -> contours
    '''
    result = []
    for cnt in contours:
        ret, contour = isQuadrilateral(cnt, esp)
        if ret:
            result.append(contour)
    return result

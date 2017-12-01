'''Helper functions for sudoku solver project'''

import numpy as np
import cv2

side_length = 400

# __sudoku_ctrl_pts3D = np.array([[[000, 000, 0]],
#                                 [[400, 000, 0]],
#                                 [[400, 400, 0]],
#                                 [[000, 400, 0]]])
__sudoku_ctrl_pts2D = np.array([[[0, 0]],
                                [[side_length, 0]],
                                [[side_length, side_length]],
                                [[0, side_length]]])


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


def getOrthophoto(img, bbox, ctrl_pts, trans="projective"):
    '''Given the image and coordinate pairs, return the orthophoto and transform

    getOrthophoto(img, bbox, ctrPts, trans) -> orthophoto, trans
    '''
    trans = cv2.getPerspectiveTransform(
        bbox.astype(np.float32), ctrl_pts.astype(np.float32))
    return cv2.warpPerspective(img, trans, (400, 400)), trans


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

kernel = np.ones((2,2),np.uint8)
def get_cells(img_bw):
    '''Given the orthophoto, return the grid of images for recognition
    If not sufficient cells found, return None, None

    get_cells(img_BW) -> img_cells, positions
    '''
    
    # img_bw = cv2.dilate(img_bw,kernel,iterations = 1)
    # TODO: check!
    _, contours, hierarchy = cv2.findContours(
        img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filterSmallContours(contours, 400 * 400, 0.01)
    # contours = filterNonQuadrilateral(contours, 0.05)

    # cv2.drawContours(img_bw,contours, -1 , (0,255,0),4)
    cv2.imshow('orth_bw', img_bw)

    if len(contours) != 81:
        print len(contours)
        return None, None

    parts_img = []
    parts_pos = []
    for cnt in contours:
        # get the part of the imgs
        # img[y:y+h, x: x+w]
        # 0    1
        # 3    2
        cnt = findVerticesCW(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        part = img_bw[cnt[0][0][1]: cnt[2][0][1], cnt[0][0][0]: cnt[2][0][0]]
        part = cv2.bitwise_not(part)
        parts_img.append(part)
        parts_pos.append((cx, cy))

    # the return ed contour starts at the right bottom corner
    # parts.reverse()
    imgs = [[None for i in range(9)] for i in range(9)]
    poss = [[None for i in range(9)] for i in range(9)]
    for i in range(9):
        for j in range(9):
            imgs[i][j] = parts_img[80 - (i * 9 + j)]
    return imgs

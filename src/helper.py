'''Helper functions for sudoku solver project'''

import numpy as np
import cv2
from line_intersection import line_intersec

side_length = 400

# __sudoku_ctrl_pts3D = np.array([[[000, 000, 0]],
#                                 [[400, 000, 0]],
#                                 [[400, 400, 0]],
#                                 [[000, 400, 0]]])
sudoku_ctrl_pts2D = np.array([[[0, 0]],
                              [[side_length, 0]],
                              [[side_length, side_length]],
                              [[0, side_length]]])


def distance(vec1, vec2):
    '''Find the euclidian distance between two points'''
    return np.linalg.norm(vec1 - vec2)


def simplify_contour(contour, eps=0.1):
    '''Given a numpy array of points (contour), return the simplifed contour

    simplifyContour(contour, eps) -> contour'''
    epsilon = eps * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def is_quadrilateral(contour, eps=0.1):
    '''Given a numpy array of points (contour), return whether the contour is a quadrilateral,
    and the simplified contour

    isQuadrilateral(contour, eps) -> isQuad, contour'''
    if len(contour) == 4:
        return True, contour
    contour = simplify_contour(contour, eps)
    if len(contour) == 4:
        return True, contour
    return False, contour


def find_vertices_cw(contour):
    '''Given a numpy array of points (contour), return a numpy array of four extreme vertices
    of shape Nx1(2channel).
    The order of the returned data would be clockwise, starting from top-left corner.

    findVerticesCW(contour) -> verticesCW
    '''
    # if verticesCW -> contourArea pos
    # elif verticesCCW -> contourArea neg
    area = cv2.contourArea(contour, True)
    if area < 0:  # CCW
        contour = np.flip(contour, axis=0)

    # find the top_left corner
    min_x = min_y = 999999
    for pt in contour:
        try:
            assert len(pt) == 1 and len(pt[0]) == 2
        except AssertionError:
            print 'The given data is not a valid contour. Nx1 2channel data expected.'
            exit(1)
        x = pt[0][0]
        y = pt[0][1]
        min_x = x if min_x > x else min_x
        min_y = y if min_y > y else min_y
    tl = np.array([min_x, min_y])

    tl_idx = 0
    min_dist = 9999999
    for i in range(len(contour)):
        dist = distance(tl, contour[i][0])
        if min_dist > dist:
            tl_idx = i
            min_dist = dist
    result = np.roll(contour, -1 * tl_idx, axis=0)
    return result


def get_orthophoto(img, bbox, ctrl_pts):
    '''Given the image and coordinate pairs, return the orthophoto and transform

    getOrthophoto(img, bbox, ctrPts, trans) -> orthophoto, trans
    '''
    trans = cv2.getPerspectiveTransform(
        bbox.astype(np.float32), ctrl_pts.astype(np.float32))
    return cv2.warpPerspective(img, trans, (400, 400)), trans


def filter_small_contours(contours, area, thresh=0.1):
    '''Given a list of contours, filter out the contours that is smaller than
    area*thresh. Return the qulified contours

    filterSmallContours(contours, area, thresh) -> contours
    '''
    return [x for x in contours if cv2.contourArea(x) > area * thresh]


def filter_non_quadrilateral(contours, esp=0.1):
    '''Given a list of contours, filter out the contours that is no like a
    quadrilateral shape. The returned contours are simplified.

    filterNonQuadrilateral(contours, esp) -> contours
    '''
    result = []
    for cnt in contours:
        ret, contour = is_quadrilateral(cnt, esp)
        if ret:
            result.append(contour)
    return result


def find_sudoku_board(img, image_type='raw'):
    '''Given the picture, try to find the sudoku board

    img: img source

    image_type: can be raw, gray. Default raw.

    find_sudoku_board(img) -> img_gray, contoursCW
    '''

    if image_type == 'raw':
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif image_type == 'gray':
        img_gray = img
    else:
        print 'Unknown image type', image_type
        exit(1)

    img_canny = cv2.Canny(img_gray, 100, 200)
    _, contours, _ = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = img_canny.shape[0] * img_canny.shape[1]
    # ignore contours that are too small:
    # define too small: the max area of the contour is smaller than 10% of the
    # screen area
    contours = filter_small_contours(contours, area, thresh=0.1)
    # ignore contours looks no where like a quadrilateral
    contours = filter_non_quadrilateral(contours, esp=0.05)
    return img_gray, [find_vertices_cw(cnt) for cnt in contours]


def draw_hough_lines(img, lines, color=(0, 0, 255), thickness=2):
    '''Given the hough lines and the src img, output the image with line drawn

    draw_hough_lines(img, lines) -> img_w_lines
    '''
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
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def get_perpendicular_lines(lines):
    '''Given the hough lines, determine the the 2 mode of the lines, which is
    essenstully the perpendicular class of the lines. Then return the lines belong to
    those groups. Horizontal lines are the group with smaller theta

    get_perpendicular_lines(lines) -> horiz_lines, verti_lines
    '''
    thetas = [line[0][1] for line in lines]
    # TODO: maybe auto is not the best solution
    freq, bins = np.histogram(thetas, bins='auto')

    # make sure the range is between 0 to 0.5*pi, otherwise the
    # following breaks. (0.9999*pi and 0 are essentialy the same angle)
    try:
        assert(bins[-1] < 0.8 * np.pi)
    except AssertionError:
        print bins[-1]

    max1_freq = 0
    idx1 = -1
    # find the first mode
    for i in range(len(freq)):
        if freq[i] > max1_freq:
            max1_freq = freq[i]
            idx1 = i
    # find the second mode
    max2_freq = 0
    idx2 = -1
    for i in range(len(freq)):
        if freq[i] > max2_freq and i != idx1:
            max2_freq = freq[i]
            idx2 = i
    # if any of the idx is -1, which means only one mode found.
    # Then just return None since we can't really get horiz and verti lines.
    if idx1 == -1 or idx2 == -1:
        # print freq, bins
        return None, None

    if idx1 < idx2:
        hori = idx2
        vert = idx1
    else:
        hori = idx1
        vert = idx2

    # TODO: verify the two modes are indeed perendicular
    # get lines
    hori_min = bins[hori]
    hori_max = bins[hori + 1]
    vert_min = bins[vert]
    vert_max = bins[vert + 1]
    horiz_lines = []
    verti_lines = []
    for line in lines:
        theta = line[0][1]
        if theta >= hori_min and theta <= hori_max:
            horiz_lines.append(line)
        elif theta >= vert_min and theta <= vert_max:
            verti_lines.append(line)
    return horiz_lines, verti_lines


def sort_lines(lines):
    '''Sort the lines according to the distance from origin (Rho)'''
    def __sort_rho(line):
        return line[0][0]
    list.sort(lines, key=__sort_rho)
    return lines


def remove_redundant_lines(lines):
    '''Given lines in one direction, remove the lines that are too close to
    each other. THIS FUNCTION SHOULD BE CALLED WITH LINES IN ONE DIRECTION ONLY

    remove_redundant_lines(lines) -> reduced_lines, spacing
    '''
    if len(lines) < 2:
        return lines, 0

    rhos = [line[0][0] for line in lines]
    spacings = [rhos[i] - rhos[i - 1] for i in range(1, len(rhos))]

    # using a simple threshold
    exp_cell_length = float(side_length) / 9
    thresh = 0.8 * exp_cell_length

    reduced_lines = []
    total_spacing = 0
    for i in range(len(spacings)):
        if spacings[i] >= thresh:
            total_spacing += spacings[i]
            reduced_lines.append(lines[i])
    reduced_lines.append(lines[-1])

    # freq, bins = np.histogram(spacings, bins='fd')

    # max_freq = 0
    # idx = 0
    # for i in range(len(freq)):
    #     if freq[i] > max_freq:
    #         max_freq = freq[i]
    #         idx = i

    # reduced_lines = []
    # total_spacing = 0
    # spacing_min = bins[idx]
    # spacing_max = bins[idx + 1]
    # for i in range(len(spacings)):
    #     if spacings[i] >= spacing_min and spacings[i] <= spacing_max:
    #         total_spacing += spacings[i]
    #         reduced_lines.append(lines[i])
    # reduced_lines.append(lines[-1])

    return reduced_lines, total_spacing / len(spacings)


def find_intersections(horiz_lines, verti_lines):
    '''Given the horizontal lines and vertical lines, find all intersections
    This function is copied from findCheckerBoard, which returns the same output:

    xIntersections: x coord of intersection of hi and vi

    yIntersections: y coord of intersection of hi and vi

    find_intersections(hl, vl) -> xIntersections, yIntersections
    '''
    n1 = len(horiz_lines)
    n2 = len(verti_lines)
    x_intersec = np.zeros((n1, n2), dtype=int)
    y_intersec = np.zeros((n1, n2), dtype=int)

    for hi in range(len(horiz_lines)):
        for vi in range(len(verti_lines)):
            x, y = line_intersec(horiz_lines[hi], verti_lines[vi])
            x_intersec[hi][vi] = x
            y_intersec[hi][vi] = y
    return x_intersec, y_intersec


def draw_intersections(img, x_intersec, y_intersec, radius=2, color=(0, 255, 0), thickness=1):
    '''Visualize the intersections. Draw circle on each intersection'''
    counter = 0
    for i in range(len(x_intersec)):
        for j in range(len(x_intersec[0])):
            counter += 1
            pos = (x_intersec[i][j], y_intersec[i][j])
            cv2.circle(img, pos, radius, color, thickness)
            cv2.putText(img, str(
                counter), (pos[0], pos[1] + 10), cv2.FONT_HERSHEY_COMPLEX, 1, color, thickness)
    # print counter
    return img


def get_cells(img_bw, rho=1, theta=5 * np.pi / 180):
    '''Given the orthophoto, return the grid of images for recognition
    If not found, return None, None.

    Method used: Hough transform to locate each grid cell

    get_cells(img_BW) -> img_cells, positions
    '''
    # TODO: could use multiple length_thresh to improve stability
    length_thresh = int(img_bw.shape[0] * 0.65)
    lines = cv2.HoughLines(img_bw, rho, theta, length_thresh)
    if lines is None:
        return None, None

    horiz_lines, verti_lines = get_perpendicular_lines(lines)
    if not horiz_lines or not verti_lines:
        return None, None

    horiz_lines = sort_lines(horiz_lines)
    verti_lines = sort_lines(verti_lines)

    horiz_lines, y_spacing = remove_redundant_lines(horiz_lines)
    verti_lines, x_spacing = remove_redundant_lines(verti_lines)

    x_intersec, y_intersec = find_intersections(horiz_lines, verti_lines)

    # get the part of the imgs
    # img[y:y+h, x: x+w]

    cells = [[None for _ in range(9)] for _ in range(9)]
    positions = [[None for _ in range(9)] for _ in range(9)]
    pad = 5

    if len(horiz_lines) == 10 and len(verti_lines) == 10:
        # under this 'perfect' situation, let's assume every line is the
        # intended line
        for hi in range(9):
            for vi in range(9):
                tl = (x_intersec[hi][vi], y_intersec[hi][vi])
                br = (x_intersec[hi + 1][vi + 1], y_intersec[hi + 1][vi + 1])
                d_h = br[0] - tl[0]
                d_v = br[1] - tl[1]
                part = img_bw[tl[1] + pad: tl[1] + d_v - pad,
                              tl[0] + pad: tl[0] + d_h - pad]
                cells[hi][vi] = part
                positions[hi][vi] = ((tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2)
        return cells, positions

    # TODO: deal with imperfect situation
    return None, None

def make_solution_img(sol):
    '''Given the sudoku solution, return a image of that'''


def get_cells_ABANDONED(img_bw):
    '''Given the orthophoto, return the grid of images for recognition
    If not sufficient cells found, return None, None

    get_cells(img_BW) -> img_cells, positions
    '''

    # img_bw = cv2.dilate(img_bw,kernel,iterations = 1)
    # TODO: check!
    _, contours, hierarchy = cv2.findContours(
        img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = filter_small_contours(contours, 400 * 400, 0.01)
    # contours = filterNonQuadrilateral(contours, 0.05)

    # cv2.drawContours(img_bw,contours, -1 , (0,255,0),4)
    cv2.imshow('orth_bw', img_bw)

    if len(contours) != 81:
        # print len(contours)
        return None, None

    parts_img = []
    parts_pos = []
    for cnt in contours:
        # get the part of the imgs
        # img[y:y+h, x: x+w]
        # 0    1
        # 3    2
        cnt = find_verticesCW(cnt)
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


def find_verticesCW_ABANDONED(contour):
    '''Given a numpy array of points (contour), return a numpy array of four extreme vertices of the shape Nx1(2channel).
    The order of the returned data would be clockwise, starting from top-left corner. For performance reason,
    this function should only be called with a contour that has few control points

    findVerticesCW(contour) -> verticesCW
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

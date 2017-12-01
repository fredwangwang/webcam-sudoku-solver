import multiprocessing
import numpy as np
import cv2
import helper
import sudoku_solver

cap = cv2.VideoCapture(0)
capWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
capHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
capArea = capWidth * capHeight

font = cv2.FONT_HERSHEY_SIMPLEX

print capWidth, capHeight, capArea

while(True):
    ret, img_raw = cap.read()
    if not ret:
        print "no frame data"
        exit(1)

    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # edge detection and find contours
    # TODO: canny is an overkill, consider change canny to adaptiveThresh
    img_canny = cv2.Canny(img_gray, 100, 200)
    _, contours, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ignore contours that are too small:
    # define too small: the max area of the contour is smaller than 10% of the screen area
    contours = helper.filterSmallContours(contours, capArea, thresh=0.1)
    # ignore contours looks no where like a quadrilateral
    contours = helper.filterNonQuadrilateral(contours, esp=0.05)

    if contours:
        cv2.drawContours(img_raw, contours, -1, (0, 150, 0), 2)
        if len(contours) > 1:
            print "multiple"
        for contour in contours:

            contour = helper.findVerticesCW(contour)

            ortho_raw, trans = helper.getOrthophoto(
                img_raw, contour, helper.__sudoku_ctrl_pts2D)
            ortho_gray = cv2.cvtColor(ortho_raw, cv2.COLOR_RGB2GRAY)
            thresh, ortho_bw = cv2.threshold(ortho_gray, 127, 255,
                                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            

            # ortho_raw = cv2.warpPerspective(
            #     img_raw, trans, (helper.side_length, helper.side_length))
            # _, cnts, _ = cv2.findContours(
            #     ortho_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # get cell tiles
            cells_img, cells_pos = helper.get_cells(ortho_bw)
            # if not cells_img:
            #     # not correctly captured
            #     continue
            # print 'detected'

            # recognize digit
            # TODO!
            cells_digit = sudoku_solver.__instance

            # solve sudoku
            sol = sudoku_solver.solve(cells_digit)

            # create a solution img
            sol_img = np.zeros((400, 400, 3), np.uint8)
            for y in range(9):
                for x in range(9):
                    if sol[y][x] != 0:
                        pos = cells_pos[y][x]
                        cv2.putText(sol_img, str(
                            sol[y][x]), pos, font, 1, (255, 255, 255), 2)

            # project the sol_img to world coords

            # cnts = helper.filterNonQuadrilateral(cnts, 0.02)
            # cv2.drawContours(ortho_raw, cnts, -1, (0, 150, 0), 2)

            # draw the corner label of the sudoku board
            for i in range(len(contour)):
                pos = contour[i]
                cv2.putText(img_raw, str(i), map(tuple, pos)[
                    0], font, 1, (255, 0, 0), 2)

            cv2.imshow('sol_img', sol_img)
            cv2.imshow('ortho', ortho_raw)
    cv2.imshow('frame', img_raw)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

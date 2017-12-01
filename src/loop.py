# import multiprocessing
import numpy as np
import cv2
import helper
from helper import *
from helper import find_sudoku_board
from sudoku_solver import solve_sudoku, SAMPLE_INSTANCE

cap = cv2.VideoCapture(0)
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capArea = capWidth * capHeight
font = cv2.FONT_HERSHEY_SIMPLEX

print capWidth, capHeight

while(True):
    ret, img_raw = cap.read()
    if not ret:
        print "no frame data"
        exit(1)

    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    _, contours = find_sudoku_board(img_gray, image_type='gray')
   
    if contours:
        if len(contours) > 1:
            print "multiple"
        for contour in contours:

            contour = helper.find_verticesCW(contour)

            ortho_raw, trans = helper.getOrthophoto(
                img_raw, contour, helper.sudoku_ctrl_pts2D)
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
            cells_digit = SAMPLE_INSTANCE
            # print cells_digit
            # solve sudoku
            sol = solve_sudoku(cells_digit)

            # create a solution img
            sol_img = np.zeros((side_length, side_length, 3), np.uint8)
            for y in range(9):
                for x in range(9):
                    if cells_digit[y][x] == 0:
                        cell_length = side_length/9
                        pos = (cell_length*x + 10,cell_length*y + 33)
                        cv2.putText(sol_img, str(
                            sol[y][x]), pos, font, 1, (255,255,255), 2)

            # project the sol_img to world coords
            inv_trans = np.linalg.inv(trans)
            sol_img_w = cv2.warpPerspective(sol_img, inv_trans, (capWidth, capHeight))
            sol_img_w_col = sol_img_w * np.array([0,0.5,0])
            mask = cv2.bitwise_not(sol_img_w)
            img_raw = cv2.bitwise_and(img_raw, mask)

            
            # img_raw = cv2.bitwise_or(img_raw, sol_img_w_col.astype(np.uint8))
            # cnts = helper.filterNonQuadrilateral(cnts, 0.02)
            # cv2.drawContours(ortho_raw, cnts, -1, (0, 150, 0), 2)

            # draw the corner label of the sudoku board
            for i in range(len(contour)):
                pos = contour[i]
                cv2.putText(img_raw, str(i), map(tuple, pos)[
                    0], font, 1, (255, 0, 0), 2)
            cv2.drawContours(img_raw, [contour], -1, (0, 150, 0), 2)
            cv2.imshow('sol_img', sol_img_w)
            cv2.imshow('ortho', ortho_raw)
    cv2.imshow('frame', img_raw)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

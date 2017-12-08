import numpy as np
import cv2
from sudoku_solver import solve_sudoku
from digit_recognition import recognize, recognize_grid
from helper import find_sudoku_board, get_orthophoto, side_length, sudoku_ctrl_pts2D, \
    get_cells, draw_hough_lines, get_perpendicular_lines, find_intersections, \
    draw_intersections, sort_lines, remove_redundant_lines


cap = cv2.VideoCapture(0)
capWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
capHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
capArea = capWidth * capHeight
font = cv2.FONT_HERSHEY_DUPLEX
kernel = np.ones((2, 2), np.uint8)
x_offset = -10
y_offset = +10

print capWidth, capHeight


def preprocessing_grid(ortho_gray):
    '''Given the gray-scale grid orthophoto, return processed ortho_bw

    preprocessing_grid(ortho_gray) -> ortho_bw
    '''
    ortho_bw = cv2.adaptiveThreshold(
        ortho_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    ortho_bw = cv2.morphologyEx(ortho_bw, cv2.MORPH_CLOSE, kernel)
    return ortho_bw


while(True):
    ret, img_raw = cap.read()
    if not ret:
        print "no frame data"
        exit(1)

    img_gray, contours = find_sudoku_board(img_raw, image_type='raw')

    for contour in contours:
        ortho_gray, trans = get_orthophoto(
            img_gray, contour, sudoku_ctrl_pts2D)
        ortho_raw = cv2.warpPerspective(
            img_raw, trans, (side_length, side_length))
        ortho_bw = preprocessing_grid(ortho_gray)

        cells, positions = get_cells(ortho_bw)
        if not cells:
            continue

        # debug
        cells_digit = [[None for _ in range(9)] for _ in range(9)]
        cells_digit = recognize_grid(cells)
        for i in range(9):
            for j in range(9):
                # cells_digit[i][j] = recognize(cells[i][j])
                cv2.putText(ortho_raw, str(cells_digit[i][j]),
                            positions[i][j], font, 0.3, (0, 255, 0), 1)

        cells_digit = recognize_grid(cells)
        sol = solve_sudoku(cells_digit)

        # create a solution img
        sol_img = np.zeros((side_length, side_length, 3), np.uint8)
        for y in range(9):
            for x in range(9):
                if cells_digit[y][x] == 0:
                    pos = positions[y][x]
                    cv2.putText(sol_img, str(
                        sol[y][x]), (pos[0]+x_offset, pos[1]+y_offset), font, 1, (255, 255, 255), 2)

        # project the sol_img to world coords
        inv_trans = np.linalg.inv(trans)
        sol_img_w = cv2.warpPerspective(
            sol_img, inv_trans, (capWidth, capHeight))
        sol_img_w_col = sol_img_w * np.array([0, 0.5, 0])
        mask = cv2.bitwise_not(sol_img_w)
        img_raw = cv2.bitwise_and(img_raw, mask)

        # draw the contour
        cv2.drawContours(img_raw, [contour], -1, (0, 255, 0), 2)

        cv2.imshow('ortho_raw', ortho_raw)
        cv2.imshow('ortho_bw', ortho_bw)
    cv2.imshow('img_raw', img_raw)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

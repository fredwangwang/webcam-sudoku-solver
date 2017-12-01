import cv2
import numpy as np

import helper

img_raw = cv2.imread('ortho_raw.jpg')


img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

thresh, img_bw = cv2.threshold(img_gray, 127, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)

res = helper.get_cells(img_bw)
cv2.imshow('cell', res[8][8])


_, contours, hierarchy = cv2.findContours(
    img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# format for hierarchy:
#     [Next, Previous, First_Child, Parent]

# helper.filt

i = 0
contours = helper.filterSmallContours(contours, 400 * 400, 0.01)
contours = helper.filterNonQuadrilateral(contours, 0.05)
print len(contours)

# cv2.drawContours(img_raw, contours, -1, (0,150,0), 2)
for cnt in contours:
    # if i == 81:
    #     break
    i += 1
    # assert(cv2.contourArea(cnt, True) < 0) # debug
    cnt = helper.findVerticesCW(cnt)
    pos = map(tuple, cnt[0])[0]
    pos = (pos[0] + 10, pos[1] + 20)
    cv2.drawContours(img_raw, [cnt], -1, (0, 150, 0), 2)
    cv2.putText(img_raw, str(i), pos, cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 0))

    # img[y:y+h, x: x+w]
    # 0    1
    # 3    2
    # print cnt[0][0][1]
    part = img_bw[cnt[0][0][1]: cnt[2][0][1], cnt[0][0][0]: cnt[2][0][0]]
    # cv2.imshow(str(i), part)


# given the knowledge that a sudoku grid should be 9x9

solution = [[9, 0, 5, 0, 0, 0, 1, 0, 2],
            [3, 0, 8, 1, 5, 7, 6, 0, 4],
            [6, 4, 0, 3, 9, 2, 0, 8, 7],
            [1, 8, 9, 0, 2, 0, 4, 6, 5],
            [7, 3, 6, 5, 0, 8, 2, 1, 9],
            [4, 5, 2, 0, 0, 0, 8, 7, 3],
            [5, 0, 0, 9, 0, 6, 0, 2, 8],
            [0, 9, 7, 4, 8, 1, 3, 5, 0],
            [0, 6, 0, 0, 7, 0, 9, 4, 0]]

cell_size = (400 - 10) / 9
offset_x = cell_size / 2
offset_y = cell_size / 2 + 10

color_img = np.zeros((400, 400, 3), np.uint8)
black_img = np.zeros((400, 400, 3), np.uint8)
img_raw = cv2.resize(img_raw, (400, 400))

font = cv2.FONT_HERSHEY_SIMPLEX
for y in range(9):
    for x in range(9):
        if solution[y][x] != 0:
            pos = (int(x * cell_size + offset_x),
                   int(y * cell_size + offset_y))
            cv2.putText(color_img, str(
                solution[y][x]), pos, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(black_img, str(
                solution[y][x]), pos, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

mask = cv2.bitwise_not(black_img)
mask = cv2.bitwise_and(mask, img_raw)
cv2.imshow('raw', img_raw)
cv2.imshow('bin', img_bw)
cv2.imshow('cl', color_img)
cv2.imshow('bk', mask)

cv2.waitKey(0)

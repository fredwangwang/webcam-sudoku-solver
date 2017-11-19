import cv2
import numpy as np

import helper


img_raw = cv2.imread('ortho_raw.jpg')

img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)

thresh, img_bw = cv2.threshold(img_gray, 127, 255,
                                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)

_, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


print len(contours)

# format for hierarchy:
#     [Next, Previous, First_Child, Parent]

# helper.filt

i = 0
contours = helper.filterSmallContours(contours, 400*400, 0.01)

print len(contours)

# cv2.drawContours(img_raw, contours, -1, (0,150,0), 2)
for cnt in contours:
    # if i == 81:
    #     break
    i+= 1
    #assert(cv2.contourArea(cnt, True) < 0) # debug
    cnt = helper.findVerticesCW(cnt)
    pos = map(tuple, cnt[0])[0]
    pos = (pos[0] + 10, pos[1] + 20)
    cv2.drawContours(img_raw, [cnt], -1, (0,150,0),2)
    cv2.putText(img_raw, str(i), pos , cv2.FONT_HERSHEY_PLAIN,1,(200,0,0))

    # img[y:y+h, x: x+w]
    # 0    1 
    # 3    2
    # print cnt[0][0][1]
    part = img_bw[cnt[0][0][1]: cnt[2][0][1] ,cnt[0][0][0]: cnt[2][0][0]]
    cv2.imshow(str(i), part)

    
# given the knowledge that a sudoku grid should be 9x9



cv2.imshow('raw',img_raw)
cv2.imshow('bin', img_bw)

cv2.waitKey(0)

'''Recognize the digit using pretrained simple neural networks'''

import os
import numpy as np
import cv2
import nn_params
import nn_model

if not os.path.exists(os.path.join(os.curdir, 'models', nn_params.MODEL_FILE)):
    print 'No pre-trained model found, train the model first!'
    exit(1)

NET = nn_model.NeuralNetwork(nn_params.LAYER_SIZES)
NET.load(nn_params.MODEL_FILE)


def recognize(img):
    '''Given the bw image, guess the number in the image.
    If the given image is nearly an empty image, return None
    '''
    bright_count = np.sum(img)
    shape = img.shape
    if float(bright_count) / (shape[0] * shape[1]) < 0.05:
        return None
    to_be_recognize = cv2.resize(img, (28, 28))
    return NET.predict(np.reshape(to_be_recognize, (784, 1)))


def recognize_grid(cells):
    '''Given the sudoku cells, return cells of digits'''
    # TODO: do the real thing here!
    return ((0, 7, 0, 8, 6, 4, 0, 3, 0),
            (0, 2, 0, 0, 0, 0, 0, 9, 0),
            (0, 0, 1, 0, 0, 0, 5, 0, 0),
            (0, 0, 0, 7, 0, 3, 0, 0, 0),
            (0, 0, 0, 0, 4, 0, 0, 0, 0),
            (0, 0, 0, 6, 1, 9, 0, 0, 0),
            (0, 1, 4, 0, 3, 0, 7, 0, 0),
            (2, 0, 0, 0, 0, 0, 0, 0, 6),
            (8, 0, 3, 2, 0, 5, 0, 0, 1))

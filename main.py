#!/usr/bin/env python

from PIL import Image
import pytesseract
import cv2
import numpy as np
from numpy.random import randint
# from solver import solveSudoku
import os
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import sys
from copy import deepcopy

print("Press Q to exit")


def solve_sudoku(sudoku_array):
    # test
    # HIER ANJAS TEIL
    # Eingabe: Sudoku als mehrdimensionales Array (je Reihe ein Unter-Array),
    #          z.B: [[5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], ...]
    #          die Nullen entsprechen den leeren Sudoku-Feldern
    # Source: https://stackoverflow.com/a/35500280/3582159

    def output(a):
        sys.stdout.write(str(a))

    N = gridsize

    def print_field(field):
        if not field:
            output("No solution")
            return
        for i in range(N):
            for j in range(N):
                cell = field[i][j]
                if cell == 0 or isinstance(cell, set):
                    output('.')
                else:
                    output(cell)
                if (j + 1) % 3 == 0 and j < 8:
                    output(' |')

                if j != 8:
                    output(' ')
            output('\n')
            if (i + 1) % 3 == 0 and i < 8:
                output("- - - + - - - + - - -\n")

    def read(field):
        """ Read field into state (replace 0 with set of possible values) """

        state = deepcopy(field)
        for i in range(N):
            for j in range(N):
                cell = state[i][j]
                if cell == 0:
                    state[i][j] = set(range(1, 10))

        return state

    state = read(sudoku_array)

    def done(state):
        """ Are we done? """

        for row in state:
            for cell in row:
                if isinstance(cell, set):
                    return False
        return True

    def propagate_step(state):
        """ Propagate one step """

        new_units = False

        for i in range(N):
            row = state[i]
            values = set([x for x in row if not isinstance(x, set)])
            for j in range(N):
                if isinstance(state[i][j], set):
                    state[i][j] -= values
                    if len(state[i][j]) == 1:
                        state[i][j] = state[i][j].pop()
                        new_units = True
                    elif len(state[i][j]) == 0:
                        return False, None

        for j in range(N):
            column = [state[x][j] for x in range(N)]
            values = set([x for x in column if not isinstance(x, set)])
            for i in range(N):
                if isinstance(state[i][j], set):
                    state[i][j] -= values
                    if len(state[i][j]) == 1:
                        state[i][j] = state[i][j].pop()
                        new_units = True
                    elif len(state[i][j]) == 0:
                        return False, None

        for x in range(3):
            for y in range(3):
                values = set()
                for i in range(3 * x, 3 * x + 3):
                    for j in range(3 * y, 3 * y + 3):
                        cell = state[i][j]
                        if not isinstance(cell, set):
                            values.add(cell)
                for i in range(3 * x, 3 * x + 3):
                    for j in range(3 * y, 3 * y + 3):
                        if isinstance(state[i][j], set):
                            state[i][j] -= values
                            if len(state[i][j]) == 1:
                                state[i][j] = state[i][j].pop()
                                new_units = True
                            elif len(state[i][j]) == 0:
                                return False, None

        return True, new_units

    def propagate(state):
        """ Propagate until we reach a fixpoint """
        while True:
            solvable, new_unit = propagate_step(state)
            if not solvable:
                return False
            if not new_unit:
                return True

    def solve(state):
        """ Solve sudoku """

        solvable = propagate(state)

        if not solvable:
            return None

        if done(state):
            return state

        for i in range(N):
            for j in range(N):
                cell = state[i][j]
                if isinstance(cell, set):
                    for value in cell:
                        new_state = deepcopy(state)
                        new_state[i][j] = value
                        solved = solve(new_state)
                        if solved is not None:
                            return solved
                    return None

    res = solve(state)
    print_field(res)
    # Ausgabe: gelöstes Sudoku in der gleichen Struktur
    if not res:
        return sudoku_array
    return res


def detect_digit(frame):
    # HIER NINAS TEIL
    # Eingabe: Zugeschnittenes Schwarz-Weißes OpenCV Bild "frame" mit genau einer oder keiner Ziffer
    """
    # (works, but slow and no good detection (not all numbers)
    filename = os.path.abspath(str("{}_"+str(row-1)+str(col-1)+".png").format(os.getpid()))
    cv2.imwrite(filename, frame)
    # Bei Fehler muss installiert werden: "AR-Sudoku/tesseract-ocr-setup...exe" und der Installationsordner muss zum System-Path hinzugefügt werden
    text = pytesseract.image_to_string(Image.open(filename), config='outputbase digits')
    # os.remove(filename)
    print(filename, "detected:", text)
    if text == "":
        return 0
    return text
    """
    # Ausgabe: Integer der Ziffer oder 0 wenn keine Ziffer vorhanden
    return 4


def processFrame(frame):
    modifiedFrame = frame.copy()

    """
    #cut to square
    h,w,c = modifiedFrame.shape
    x = int((w-h)/2)
    modifiedFrame = modifiedFrame[0:h, x:x + h]
    """

    # convert to black and white
    modifiedFrame = cv2.cvtColor(modifiedFrame, cv2.COLOR_BGR2GRAY)
    (thresh, modifiedFrame) = cv2.threshold(modifiedFrame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # detect outlines
    blurred = cv2.GaussianBlur(modifiedFrame, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # biggest rectangle
    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    sudoku_outline = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, we have found the sudoku
        if len(approx) == 4:
            sudoku_outline = approx
            break

    if sudoku_outline is not None:
        # extract the sudoku, apply a perspective transform to it
        warped = four_point_transform(modifiedFrame, sudoku_outline.reshape(4, 2))
        # warped_on_original_frame = four_point_transform(frame, sudoku_outline.reshape(4, 2))

        # sharpen
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # warped = cv2.filter2D(warped, -1, kernel)

        # if input image is big, resize: 500px are enough and resize removes some noise
        if warped.shape[0] > process_size:
            warped = cv2.resize(warped, (process_size, process_size))

        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        """
        #horizontale Linien entfernen
        linek = np.zeros((18,18),dtype=np.uint8)
        linek[9,...]=1
        x=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, linek ,iterations=1)
        thresh2 = thresh - x
        cv2.imshow("x", x)
        cv2.imshow("test3", thresh2)
        """

        # find small squares
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        thresh_draw = thresh.copy()
        thresh_draw = cv2.cvtColor(thresh_draw, cv2.COLOR_GRAY2BGR)

        warped_h, warped_w = warped.shape
        rowh = int(warped_h / gridsize)
        colw = int(warped_w / gridsize)

        # squares = np.zeros((gridsize, gridsize)).tolist()
        sudoku = np.zeros((gridsize, gridsize), dtype=np.uint).tolist()
        squares_pos = np.zeros((gridsize, gridsize)).tolist()  # tupels (x,y,w,h)

        # loop over the contours
        # Alternativ in 9x9 grid einteilen statt kleine Kästchen suchen
        if useContours:
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                # only use if the contour has four vertices and if its width is smaller than twice the grid cell size but bigger than 1/18 of the grid size
                if len(approx) == 4:
                    x, y = approx[0][0]
                    # print(approx, len(approx))
                    w = approx[2][0][0] - x
                    h = approx[2][0][1] - y
                    if colw / 18 < w < colw * 2:
                        # split image in 9 rows and cols and determine which the contour is most likely in
                        col_nr = 0
                        row_nr = 0
                        for i in range(1, gridsize + 1):
                            if (i - 1) * colw <= x + int(colw / 2) < i * colw:  # Spalte bestimmen
                                col_nr = i
                            if (i - 1) * rowh <= y + int(rowh / 2) < i * rowh:  # Reihe bestimmen
                                row_nr = i
                        squares_pos[row_nr - 1][col_nr - 1] = (x, y, w, h)
                        i += 1
        else:  # divide grid 9x9:
            for row_nr in range(0, gridsize):
                y = row_nr * rowh
                for col_nr in range(0, gridsize):
                    x = col_nr * colw
                    squares_pos[row_nr][col_nr] = (x, y, colw, rowh)

        """
            Scan Numbers
        """
        # do something for each detected square
        for row_nr in range(0, gridsize):
            for col_nr in range(0, gridsize):
                if type(squares_pos[row_nr][col_nr]) is not float:
                    x, y, w, h = squares_pos[row_nr][col_nr]
                    cutted_square = warped[y:y + h, x:x + w]
                    sudoku[row_nr][col_nr] = detect_digit(cutted_square)
                    # debug: show border and colum nr
                    color = (randint(0, 255), randint(0, 255), randint(0, 255))
                    cv2.rectangle(thresh_draw, (x, y), (x + colw, y + rowh), color, 2)
                    # cv2.circle(thresh_draw, (x + int(colw / 2), y + int(rowh / 2)), 1, color)
                    # cv2.putText(thresh_draw, str(col_nr + 1) + str(row_nr + 1), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                # else:
                #    print("skipped square scanning")

        # TEST overwrite sudoku to test solver
        # sudoku = field
        sudoku = [[5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
                 [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
                 [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9]]

        """
            Solve Sudoku
        """

        sudoku_solved = solve_sudoku(sudoku)

        # TEST overwrite sudoku to test output print
        #sudoku_solved = [[5, 3, 4, 6, 7, 8, 9, 1, 2], [6, 7, 2, 1, 9, 5, 3, 4, 8], [1, 9, 8, 3, 4, 2, 5, 6, 7],
        #                 [8, 5, 9, 7, 6, 1, 4, 2, 3], [4, 2, 6, 8, 5, 3, 7, 9, 1], [7, 1, 3, 9, 2, 4, 8, 5, 6],
        #                 [9, 6, 1, 5, 3, 7, 2, 8, 4], [2, 8, 7, 4, 1, 9, 6, 3, 5], [3, 4, 5, 2, 8, 6, 1, 7, 9]]

        """
            Print Results on Input Image
        """
        # do something for each detected square
        # print(squares_pos)
        for row_nr in range(0, gridsize):
            for col_nr in range(0, gridsize):
                if type(squares_pos[row_nr][col_nr]) is not float:
                    x, y, w, h = squares_pos[row_nr][col_nr]
                    if sudoku[row_nr][col_nr] == 0:
                        # print("printing square", str(sudoku_solved[row_nr][col_nr]))
                        cv2.putText(thresh_draw, str(sudoku_solved[row_nr][col_nr]), (x + 8, y + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(244, 255, 255), lineType=2)
                # else:
                #    print("skipped square printing")

        output = cv2.resize(thresh_draw, (process_size, process_size))

        debug = edged.copy()
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        color = (42, 173, 32)
        cv2.rectangle(debug, (sudoku_outline[0][0][0], sudoku_outline[0][0][1]),
                      (sudoku_outline[2][0][0], sudoku_outline[2][0][1]), color, 4)
        debug = cv2.resize(debug, (process_size, process_size))

    else:  # if no sudoku outline is found
        output = cv2.resize(edged, (process_size, process_size))
        debug = output

    return output, debug


def showOutput(imgs):
    if combineOutputWindow:
        cv2.imshow("Output", np.concatenate(imgs, axis=1))
    else:
        for nr, img in enumerate(imgs):
            cv2.imshow("Output " + str(nr), img)


################################
#       RUN PROGRAM            #
################################

# config
captureWebcam = True
input_file = "examples/example_5.jpg"  # example 5 and 1 work
process_size = 500
useContours = True
combineOutputWindow = True
gridsize = 9                  # 9x9 sudoku

if captureWebcam:
    cap = cv2.VideoCapture(0)

    while True:
        # read and modify video
        ret, frame = cap.read()
        if ret:
            showOutput(processFrame(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    frame = cv2.imread(input_file)
    showOutput(processFrame(frame))
    cv2.waitKey(0)

from typing import Any, Tuple, Iterator
import cv2
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
from pprint import pprint


def pre_process_image(img, save_in_file=None, morph_size=(8, 8)):
    plotting = plt.imshow(img)
    plt.show()
    pre = None
    # get rid of the color
    # pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre = cv2.convertScaleAbs(img)
    # Otsu threshold
    try :
        print('pre-pro-img--')
        pre,_ = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print('pre-pro-img--2')
        # dilate the text to make it solid spot
        cpy = pre.copy()
        struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
        cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
        pre = ~cpy

        if save_in_file is not None:
            cv2.imwrite(save_in_file, pre)
    except Exception as e:
            pprint(e)
    return pre

def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    img = pre.copy()
    cv2.drawContours(img, contours, -1, (0, 255, 255))
    plotting = plt.imshow(img)
    plt.show()
    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=2):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]
    # plotting = plt.imshow(vis, cmap='gray')
    # plt.show()
    # Filtering out the clusters having less than 2 cols
    # table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    table_cells = list(rows.values())
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    return table_cells


def get_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    print('generating vh lines')
    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    print('drawing v lines')
    ver_lines = []
    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    print('drawing h lines')
    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))

    return hor_lines, ver_lines


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    print("Table cells length", len(table_cells))
    print(table_cells)

    hor_lines = []
    ver_lines = []

    column_first = [boxes[0] for boxes in table_cells]
    column_last = [boxes[-1] for boxes in table_cells]
    print("first column",column_first)
    print("last column",column_last)
    box_x_min = min(column_first, key=itemgetter(0))
    # box_x_max = max(column_last, key=itemgetter(0))
    box_x_max = column_last[0]
    print("box_x_max", box_x_max)
    for i in range(len(column_last)):
        box_x_max_x, _, box_x_max_w, _ = box_x_max
        box_x, _, box_w, _ = column_last[i]
        if i > 0:
            prev_box_x, _, prev_box_w, _ = column_last[i - 1]
            if (box_x + box_w) > (box_x_max_x + box_x_max_w) and box_x - prev_box_x > 15:
                box_x_max = column_last[i]
        else:
            if (box_x + box_w) > (box_x_max_x + box_x_max_w):
                box_x_max = column_last[i]

    print("box_x_max", box_x_max)
    x_min,_,_,_ = box_x_min
    x_max,_,x_max_w,_ = box_x_max
    left_x = x_min
    right_x = x_max + x_max_w
    for boxes in table_cells:
        min_y_box = min(boxes, key=itemgetter(1))
        # max_y_box = max(boxes, key=itemgetter(0))
        _, y1, _, _ = min_y_box
        # _, y2, _, _ = max_y_box
        # y_min = min(y1,y2)
        print("boxes",min_y_box)
        # draw vertical lines
        # cv2.line(vis, (x_min, y1), (x_max + x_max_w , y1), (0, 255, 255), 1)
        ver_lines.append((x_min, y1, x_max + x_max_w , y1))
    # plotting = plt.imshow(vis, cmap='gray')
    # plt.show()

    row_first = table_cells[0]
    row_last = table_cells[-1]
    print("row_first", row_first)
    print("row_last", row_last)

    box_y_min = min(row_first, key=itemgetter(1))
    print("box_y_min", box_y_min)
    box_y_max = max(row_last, key=itemgetter(1))
    print("box_y_min", box_y_max)

    for i in range(len(row_last)):
        _, box_y_max_y, _, box_y_max_h = box_y_max
        _, box_y, _, box_h = row_last[i]
        _, prev_box_y, _, prev_box_h = row_last[i-1]
        if i> 0:
            if (box_y + box_h) > (box_y_max_y + box_y_max_h) and box_y - prev_box_y:
                box_y_max = row_last[i]
        else:
            if (box_y + box_h) > (box_y_max_y + box_y_max_h):
                box_y_max = row_last[i]

    columns = []
    global max_length
    for i in range(len(table_cells)):
        row = table_cells[i]
        column = []
        max_length = max([len(r) for r in table_cells])
        print("max_length", max_length)
        for j in range(max_length):
            try:
                print("row[j]", row[j])
                column.append(row[j])
            except:
                column.append([0,0,0,0])
        columns.append(column)

    print("columns", columns)
    columns = np.array(columns)
    print("columns", columns)

    _, y_min, _, _ = box_y_min
    _, y_max, _, y_max_h = box_y_max

    top_y = y_min
    bottom_y = y_max + y_max_h

    for i in range(max_length):
        boxes = columns[:, i]
        print("boxes", boxes)
        try :
            min_x_boxes = [box for box in boxes if not box[i] == 0 ]
        except:
            pass
        min_x = [box[0] for box in min_x_boxes ]
        # min_x = sorted(min_x)
        # mean_x = int(np.std(min_x))
        # std_x = int(np.std(min_x))
        # md_x = np.mean(np.absolute(min_x - np.mean(min_x)))
        # cv = cv = lambda x: np.std(x) / np.mean(x)
        # cv_x = cv(min_x)
        # var_x = int(np.var(min_x))
        minimum = min(min_x)
        # print("min_x_boxes", min_x)
        # print("min",minimum)
        # print("mean", mean_x)
        # print("std", std_x)
        # print("md", md_x)
        # print("var", var_x)
        # print("cv", cv_x)
        x1 = int(minimum)
        print("x1", x1)
        # x1, _, _, _ = min_x_box
        print("min_x_boxes", min_x_boxes)
        # cv2.line(vis, (x1, y_min), (x1, y_max + y_max_h), (255, 255, 0), 1)
        hor_lines.append((x1, y_min, x1, y_max + y_max_h))
    # cv2.line(vis, (right_x, top_y), (right_x, bottom_y), (0, 0, 255), 1)
    ver_lines.append((right_x, top_y, right_x, bottom_y))
    # cv2.line(vis, (left_x, bottom_y), (right_x, bottom_y), (0, 0, 255), 1)
    ver_lines.append((left_x, bottom_y, right_x, bottom_y))
    # plotting = plt.imshow(vis, cmap='gray')
    # plt.show()

    # for i in range(max_length):
    #     print("[%d]" %i, columns[:, i])
    #     sorted_columns = sorted(columns[:, i],key=itemgetter(0))
    #     print("sorted_columns", sorted_columns)
    #     print("sorted_columns mean x", np.mean(sorted_columns,axis=0))
    #     min_x_box = sorted_columns[0]
    #     zero = np.array([0,0,0,0])
    #     if np.array_equal(min_x_box, zero):
    #         for row in sorted_columns:
    #             if not np.array_equal(row, zero):
    #                 min_x_box = row
    #                 break
    #     # max_x_box = max(columns[:, i], key=itemgetter(1))
    #     x1, _, _, _ = min_x_box
    #     # x2, _, _, _ = max_x_box
    #     # x_min = min(x1, x2)
    #     print("boxes", min_x_box)
    #     cv2.line(vis, (x1, y_min), (x1, y_max + y_max_h), (255, 255, 0), 1)
    #     hor_lines.append((x1, y_min, x1, y_max + y_max_h))
    # plotting = plt.imshow(vis, cmap='gray')
    # plt.show()

    return hor_lines, ver_lines


def reject_outliers(data, m=2):
    mean = np.mean(data)
    std = np.std(data)
    return [x if x-mean < m*std else 0 for x in data]


def invert(image):
    # thresholding the image to a binary image
    thresh, img_bin = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # inverting the image
    img_bin = 255 - img_bin
    return img_bin


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY,dstCn=1)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# closing - erosion followed by dilation
def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 80)


# skew correction
def deskew(image):
    print("deskewing image")
    thresh = cv2.threshold(image, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    print("rotating back by angle %s" %angle)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def save_image_w_lines(iproc_obj, img_file_name):
    img_lines = iproc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = os.path.join(img_file_name)

    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
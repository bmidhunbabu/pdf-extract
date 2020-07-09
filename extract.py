import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from PIL.Image import core as image
# try:
#     from PIL import Image
# except ImportError:
#     import Image
import pytesseract
from pytesseract import Output
import sys
import os
import re

from utils import converter
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y
from math import radians, degrees
from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes
from utils import table_extractor as te
from utils import pdf_extractor as pe

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


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 80)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
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

file='E:\Projects\Pyhton\PDF Etract\input\extract.png'
pre = file[:file.rindex('.')] + '-pre.png'
out = file[:file.rindex('.')] + '-out.png'


img = cv2.imread(file)
# plotting = plt.imshow(img)
# plt.show()

# img = deskew(img)


# print("thresh")
# thresh = thresholding(img - 255)
# plotting = plt.imshow(thresh)
# plt.show()

noiss_red = remove_noise(img)
gray = get_grayscale(noiss_red)

canny = canny(gray)
plotting = plt.imshow(canny)
plt.show()
input = file[:file.rindex(".")] + "-canny.jpg"
cv2.imwrite(input,canny)

plotting = plt.imshow(cv2.imread(input))
plt.show()
inverted = file[:file.rindex(".")] + "-pre-processed.jpg"
processed = invert(cv2.imread(input,0))
cv2.imwrite(inverted,processed)

# img = te.pre_process_image(cv2.imread(inverted), inverted)

img = cv2.imread(inverted)

# h, w, c = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
#
# plotting = plt.imshow(img)
# plt.show()
# cv2.waitKey(0)

# d = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
# print(d['text'])
# print(d)
#
# n_boxes = len(d['text'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 60:
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         canny = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# # cv2.imwrite(file,canny)
# plt.show()
# cv2.waitKey(0)
# cv2.waitKey(0)




#read your file
# try:
#     pre_processed = te.pre_process_image(img, pre)
#     text_boxes = te.find_text_boxes(pre_processed)
#     cells = te.find_table_in_boxes(text_boxes,out)
#     hor_lines, ver_lines = te.build_lines(cells,out)
# except Exception as e:
#     exc_type, exc_obj, exc_tb = sys.exc_info()
#     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
#     print(str(e), "on", fname, "at line : ", exc_tb.tb_lineno)
#     print(str(e))
#
# # Visualize the result
# vis = img.copy()

# for box in text_boxes:
#     (x, y, w, h) = box
#     cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 1)
#
# for line in hor_lines:
#     [x1, y1, x2, y2] = line
#     cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
#
# for line in ver_lines:
#     [x1, y1, x2, y2] = line
#     cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
#
# cv2.imwrite(out, vis)


#read your file
img = cv2.imread(file,0)
print(img.shape)
#thresholding the image to a binary image
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
#inverting the image
img_bin = 255-img_bin
cv2.imwrite(file[:file.rindex('.')] + "-inverted.jpg",img_bin)
#Plotting the image t
plotting = plt.imshow(img_bin,cmap='gray')
plt.show()


# Length(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1]//100
print(kernel_len)
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
print(kernel)


#Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
# vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_OPEN, kernel)
cv2.imwrite(file[:file.rindex('.')] + "-vertical.jpg",vertical_lines)
#Plot the generated image
plotting = plt.imshow(vertical_lines,cmap='gray')
plt.show()


#Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
# horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, hor_kernel)
cv2.imwrite(file[:file.rindex('.')] + "-horizontal.jpg",horizontal_lines)
#Plot the generated image
plotting = plt.imshow(horizontal_lines,cmap='gray')
plt.show()


# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
#Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite(file[:file.rindex('.')] + "-vh.jpg", img_vh)
# smooth = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
bitxor = cv2.bitwise_xor(img,img_vh)
bitnot = cv2.bitwise_not(bitxor)
#Plotting the generated image
plotting = plt.imshow(bitnot,cmap='gray')
plt.show()


# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours, method = "top-to-bottom")
print(contours)
#Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
#Get mean of heights
mean = np.mean(heights)

#Create list box to store all boxes in
box = []
# Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w<1000 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        box.append([x,y,w,h])
cv2.imwrite(file[:file.rindex('.')] + "-rectangle.jpg", image)
plotting = plt.imshow(image,cmap='gray')
plt.show()

# Creating two lists to define row and column in which cell is located
row = []
column = []
j = 0

# Sorting the boxes to their respective row and column
for i in range(len(box)):

    if (i == 0):
        column.append(box[i])
        previous = box[i]

    else:
        if (box[i][1] <= previous[1] + mean / 2):
            column.append(box[i])
            previous = box[i]

            if (i == len(box) - 1):
                row.append(column)

        else:
            row.append(column)
            column = []
            previous = box[i]
            column.append(box[i])

print(column)
print(row)

# calculating maximum number of cells
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol

# Retrieving the center of each column
center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]

center = np.array(center)
center.sort()
print(center)
# Regarding the distance to the columns center, the boxes are arranged in respective order

finalboxes = []
for i in range(len(row)):
    lis = []
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)

# from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer = []
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner = ''
        if (len(finalboxes[i][j]) == 0):
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                             finalboxes[i][j][k][3]
                finalimg = bitnot[x:x + h, y:y + w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel, iterations=1)
                erosion = cv2.erode(dilation, kernel, iterations=2)

                try:
                    out = pytesseract.image_to_string(erosion,lang="eng")
                    if (len(out) == 0):
                        out = pytesseract.image_to_string(erosion, lang="eng", config='--psm 3')
                    inner = inner + " " + out
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(str(e), "on", fname, "at line : ", exc_tb.tb_lineno)
                    print(str(e))
            outer.append(inner)

# Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(" Data Frame ")
print(dataframe)
dfProperties = {
  "align": "left"
}
# data = dataframe.style.set_properties(dfProperties)
# Converting it in a excel-file
dataframe.to_excel(file[:file.rindex('.')] + ".xlsx")
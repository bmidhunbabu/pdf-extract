import os
import ntpath
import cv2
import numpy as np
from math import radians, degrees

from utils import helpers

from PyQt5.QtWidgets import QTextEdit

from pprint import pprint
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y
from pdftabextract.geom import pt
from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes
from pdftabextract.clustering import find_clusters_1d_break_dist

class PageExtractor:
    def __init__(self, page, page_no, file, output_path, xmltree):
        self.tedit = helpers.getMainWindow().findChild(QTextEdit)
        self.page = page
        self.page_no = page_no
        self.file = file
        self.OUTPUT_PATH = output_path
        self.xmltree = xmltree

    def save_page_as_image(self):
        self.tedit.append('Working on page %d' % self.page['number'])
        self.tedit.append('Width : %d px' % self.page['width'])
        self.tedit.append('Height : %d px' % self.page['height'])
        print('image %s' % self.page['image'])
        # print('the first three text boxes:')
        # pprint(p['texts'][:3])
        # get the image file of the scanned page
        self.imgfilebasename = self.page['image'][:self.page['image'].rindex('.')]
        self.imgfile = self.page['image']
        self.tedit.append("Detecting lines...")
        # create an image processing object with the scanned page
        self.iproc_obj = imgproc.ImageProc(self.imgfile)
        # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
        page_scaling_x = self.iproc_obj.img_w / self.page['width']  # scaling in X-direction
        page_scaling_y = self.iproc_obj.img_h / self.page['height']  # scaling in Y-direction
        # detect the lines
        lines_hough = self.iproc_obj.detect_lines(canny_kernel_size=3, canny_low_thresh=50, canny_high_thresh=150,
                                             hough_rho_res=1,
                                             hough_theta_res=np.pi / 500,
                                             hough_votes_thresh=round(0.2 * self.iproc_obj.img_w))
        self.tedit.append("found %d lines" % len(lines_hough))
        self.save_image_w_lines(self.imgfilebasename.split('\\')[-1])
        self.tedit.append("saving image file")


    def find_skew_rotation(self):
        # find rotation or skew
        # the parameters are:
        # 1. the minimum threshold in radians for a rotation to be counted as such
        # 2. the maximum threshold for the difference between horizontal and vertical line rotation (to detect skew)
        # 3. an optional threshold to filter out "stray" lines whose angle is too far apart from the median angle of
        #    all other lines that go in the same direction (no effect here)
        rot_or_skew_type, rot_or_skew_radians = self.iproc_obj.find_rotation_or_skew(radians(0.5),  # uses "lines_hough"
                                                                                radians(1),
                                                                                omit_on_rot_thresh=radians(0.5))

        # rotate back or deskew text boxes
        needs_fix = True
        if rot_or_skew_type == ROTATION:
            print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
            rotate_textboxes(self.page, -rot_or_skew_radians, pt(0, 0))
        elif rot_or_skew_type in (SKEW_X, SKEW_Y):
            print("> deskewing in direction '%s' by %f°" % (rot_or_skew_type, -degrees(rot_or_skew_radians)))
            deskew_textboxes(self.page, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
        else:
            needs_fix = False
            print("> no page rotation / skew found")

        # rotate back or deskew detected lines
        if needs_fix:
            lines_hough = self.iproc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)
            self.save_image_w_lines(self.imgfilebasename + '-repaired')

        # save repaired XML (i.e. XML with deskewed textbox positions)
        output_files_basename = self.file[:self.file.rindex('.')]
        repaired_xmlfile = os.path.join(self.OUTPUT_PATH, output_files_basename + '.repaired.xml')

        print("saving repaired XML file to '%s'..." % repaired_xmlfile)
        self.tedit.append("Saved modified XML file to '%s'..." % ntpath.abspath(repaired_xmlfile))
        self.xmltree.write(repaired_xmlfile)


    def save_image_w_lines(self, img_file_name):
        img_lines = self.iproc_obj.draw_lines(orig_img_as_background=True)
        img_lines_file = os.path.join(self.OUTPUT_PATH, '%s-lines-orig.png' % img_file_name)

        print("> saving image with detected lines to '%s'" % img_lines_file)
        cv2.imwrite(img_lines_file, img_lines)
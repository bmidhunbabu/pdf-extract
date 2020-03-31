import os
import ntpath
import cv2
import numpy as np

from PyQt5.QtWidgets import QTextEdit

from pdftabextract.common import read_xml, parse_pages
from pprint import pprint
from pdftabextract import imgproc

from utils import helpers

class Extractor:
    def __init__(self, file):
        self.file = ntpath.basename(file)
        self.filepath_rel = ntpath.relpath(file)
        self.OUTPUT_PATH = os.path.join(ntpath.dirname(self.filepath_rel),'generated_output')
        print(self.OUTPUT_PATH)
        if not os.path.exists(self.OUTPUT_PATH) :
            os.mkdir(self.OUTPUT_PATH)
            print('dir created')


    def save_image_w_lines(self, iproc_obj, imgfilebasename):
        img_lines = iproc_obj.draw_lines(orig_img_as_background=True)
        img_lines_file = os.path.join(self.OUTPUT_PATH, '%s-lines-orig.png' % imgfilebasename)

        print("> saving image with detected lines to '%s'" % img_lines_file)
        cv2.imwrite(img_lines_file, img_lines)


    def parseXML(self):
        tedit = helpers.getMainWindow().findChild(QTextEdit)
        # Load the XML that was generated with pdftohtml
        xmltree, xmlroot = read_xml(self.filepath_rel)
        # parse it and generate a dict of pages
        pages = parse_pages(xmlroot)
        tedit.append('parssing XML file...')
        p_num = 3
        p = pages[p_num]
        tedit.append('Detected %d pages' % p['number'])
        tedit.append('Width : %d px' % p['width'])
        tedit.append('Height : %d px' % p['height'])
        print('image %s' % p['image'])
        # print('the first three text boxes:')
        # pprint(p['texts'][:3])
        # get the image file of the scanned page
        imgfilebasename = p['image'][:p['image'].rindex('.')]
        imgfile = p['image']
        tedit.append("page %d: detecting lines..." % p_num)
        # create an image processing object with the scanned page
        iproc_obj = imgproc.ImageProc(imgfile)
        # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
        page_scaling_x = iproc_obj.img_w / p['width']  # scaling in X-direction
        page_scaling_y = iproc_obj.img_h / p['height']  # scaling in Y-direction
        # detect the lines
        lines_hough = iproc_obj.detect_lines(canny_kernel_size=3, canny_low_thresh=50, canny_high_thresh=150,
                                             hough_rho_res=1,
                                             hough_theta_res=np.pi / 500,
                                             hough_votes_thresh=round(0.2 * iproc_obj.img_w))
        tedit.append("found %d lines" % len(lines_hough))
        self.save_image_w_lines(iproc_obj, imgfilebasename.split('\\')[-1])
        tedit.append("saving image file")

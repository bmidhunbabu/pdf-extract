import ntpath
import os
import cv2
from pdftabextract.common import read_xml, parse_pages
from utils import converter
import numpy as np
from pdftabextract import imgproc
from math import radians, degrees
from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y
from pdftabextract.geom import pt
from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes
from pdftabextract.clustering import find_clusters_1d_break_dist


def pre_process(pdf_file):
    DATAPATH = ntpath.dirname(pdf_file)
    HEADER_ROW_HEIGHT = 90
    xml_file = converter.toXML(pdf_file)
    # Load the XML that was generated with pdftohtml
    xmltree, xmlroot = read_xml(xml_file)
    # parse it and generate a dict of pages
    pages = parse_pages(xmlroot)
    p_num = 1
    p = pages[p_num]
    # get the image file of the scanned page
    imgfilebasename = p['image'][:p['image'].rindex('.')]
    imgfile = p['image']

    print("page %d: detecting lines in image file '%s'..." % (p_num, imgfile))

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
    print("> found %d lines" % len(lines_hough))
    rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(radians(0.5),  # uses "lines_hough"
                                                                            radians(1),
                                                                            omit_on_rot_thresh=radians(0.5))

    # rotate back or deskew text boxes
    needs_fix = True
    if rot_or_skew_type == ROTATION:
        print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
        rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
    elif rot_or_skew_type in (SKEW_X, SKEW_Y):
        print("> deskewing in direction '%s' by %f°" % (rot_or_skew_type, -degrees(rot_or_skew_radians)))
        deskew_textboxes(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
    else:
        needs_fix = False
        print("> no page rotation / skew found")

    if needs_fix:
        # rotate back or deskew detected lines
        lines_hough = iproc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)

        save_image_w_lines(iproc_obj, imgfilebasename + '-repaired')

    hori_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_HORIZONTAL, find_clusters_1d_break_dist,
                                            dist_thresh=HEADER_ROW_HEIGHT / 2)

    if len(hori_clusters) > 0:
        # draw the clusters
        img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_HORIZONTAL, hori_clusters)
        save_img_file = '%s-hori-clusters.png' % imgfilebasename
        cv2.imwrite(save_img_file, img_w_clusters)

    MIN_COL_WIDTH = 60
    vertical_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                                remove_empty_cluster_sections_use_texts=p['texts'],
                                                remove_empty_cluster_sections_n_texts_ratio=0.1,
                                                remove_empty_cluster_sections_scaling=page_scaling_x,
                                                dist_thresh=MIN_COL_WIDTH / 2)
    print("> found %d clusters" % len(vertical_clusters))

    # draw the clusters
    img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
    save_img_file = '%s-vertical-clusters.png' % imgfilebasename
    print("> saving image with detected vertical clusters to '%s'" % save_img_file)
    cv2.imwrite(save_img_file, img_w_clusters)


def save_image_w_lines(iproc_obj, imgfilebasename):
    img_lines = iproc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = '%s-lines-orig.png' % imgfilebasename

    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)
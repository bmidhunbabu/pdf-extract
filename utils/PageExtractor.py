import os
import ntpath
import cv2
import re
import numpy as np
from math import radians, degrees

from utils import helpers

from PyQt5.QtWidgets import QTextEdit

from pprint import pprint
from pdftabextract import imgproc
from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y,\
    all_a_in_b, DIRECTION_VERTICAL, \
    save_page_grids
from pdftabextract.geom import pt
from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes
from pdftabextract.clustering import find_clusters_1d_break_dist, \
    calc_cluster_centers_1d,\
    zip_clusters_and_values
from pdftabextract.textboxes import border_positions_from_texts, \
    split_texts_by_positions, join_texts
from pdftabextract.extract import make_grid_from_positions,\
    fit_texts_into_grid, datatable_to_dataframe


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
        self.page_scaling_x = self.iproc_obj.img_w / self.page['width']  # scaling in X-direction
        self.page_scaling_y = self.iproc_obj.img_h / self.page['height']  # scaling in Y-direction
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


    def find_vertical_clusters(self):
        MIN_COL_WIDTH = 60  # minimum width of a column in pixels, measured in the scanned pages

        # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
        # (break on distance MIN_COL_WIDTH/2)
        # additionaly, remove all cluster sections that are considered empty
        # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
        # per cluster section
        vertical_clusters = self.iproc_obj.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                                    remove_empty_cluster_sections_use_texts=self.page['texts'],
                                                    # use this page's textboxes
                                                    remove_empty_cluster_sections_n_texts_ratio=0.1,  # 10% rule
                                                    remove_empty_cluster_sections_scaling=self.page_scaling_x,
                                                    # the positions are in "scanned image space" -> we scale them to "text box space"
                                                    dist_thresh=MIN_COL_WIDTH / 2)
        print("> found %d clusters" % len(vertical_clusters))
        self.tedit.append("Found %d clusters" % len(vertical_clusters))

        # draw the clusters
        img_with_clusters = self.iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
        save_img_file = os.path.join(self.OUTPUT_PATH, '%s-vertical-clusters.png' % self.imgfilebasename.split('\\')[-1])
        print("> saving image with detected vertical clusters to '%s'" % save_img_file)
        cv2.imwrite(save_img_file, img_with_clusters)

        self.page_col_pos = np.array(calc_cluster_centers_1d(vertical_clusters)) / self.page_scaling_x
        print('found %d column borders:' % len(self.page_col_pos))
        print(self.page_col_pos)


    def find_horizontal_clusters(self):
        # right border of the second column
        self.col2_rightborder = self.page_col_pos[2]

        # calculate median text box height
        self.median_text_height = np.median([t['height'] for t in self.page['texts']])

        # get all texts in the first two columns with a "usual" textbox height
        # we will only use these text boxes in order to determine the line positions because they are more "stable"
        # otherwise, especially the right side of the column header can lead to problems detecting the first table row
        self.text_height_deviation_thresh = self.median_text_height / 2
        self.texts_cols_1_2 = [t for t in self.page['texts']
                          if t['right'] <= self.col2_rightborder
                          and abs(t['height'] - self.median_text_height) <= self.text_height_deviation_thresh]

        # get all textboxes' top and bottom border positions
        borders_y = border_positions_from_texts(self.texts_cols_1_2, DIRECTION_VERTICAL)

        # break into clusters using half of the median text height as break distance
        clusters_y = find_clusters_1d_break_dist(borders_y, dist_thresh=self.median_text_height / 2)
        clusters_w_vals = zip_clusters_and_values(clusters_y, borders_y)

        # for each cluster, calculate the median as center
        self.pos_y = calc_cluster_centers_1d(clusters_w_vals)
        self.pos_y.append(self.page['height'])

        print('number of line positions:', len(self.pos_y))


    def find_header_and_footer(self):
        # a (possibly malformed) population number + space + start of city name
        pttrn_table_row_beginning = re.compile(r'^[\d Oo][\d Oo]{2,} +[A-ZÄÖÜ]')

        # 1. try to find the top row of the table
        texts_cols_1_2_per_line = split_texts_by_positions(self.texts_cols_1_2, self.pos_y, DIRECTION_VERTICAL,
                                                           alignment='middle',
                                                           enrich_with_positions=True)

        # go through the texts line per line
        for line_texts, (line_top, line_bottom) in texts_cols_1_2_per_line:
            line_str = join_texts(line_texts)
            if pttrn_table_row_beginning.match(line_str):  # check if the line content matches the given pattern
                top_y = line_top
                break
        else:
            top_y = 0

        # hints for a footer text box
        words_in_footer = ('anzeige', 'annahme', 'ala')

        # 2. try to find the bottom row of the table
        min_footer_text_height = self.median_text_height * 1.5
        min_footer_y_pos = self.page['height'] * 0.7
        # get all texts in the lower 30% of the page that have are at least 50% bigger than the median textbox height
        bottom_texts = [t for t in self.page['texts']
                        if t['top'] >= min_footer_y_pos and t['height'] >= min_footer_text_height]
        bottom_texts_per_line = split_texts_by_positions(bottom_texts,
                                                         self.pos_y + [self.page['height']],  # always down to the end of the page
                                                         DIRECTION_VERTICAL,
                                                         alignment='middle',
                                                         enrich_with_positions=True)
        # go through the texts at the bottom line per line
        page_span = self.page_col_pos[-1] - self.page_col_pos[0]
        min_footer_text_width = page_span * 0.8
        for line_texts, (line_top, line_bottom) in bottom_texts_per_line:
            line_str = join_texts(line_texts)
            has_wide_footer_text = any(t['width'] >= min_footer_text_width for t in line_texts)
            # check if there's at least one wide text or if all of the required words for a footer match
            if has_wide_footer_text or all_a_in_b(words_in_footer, line_str):
                bottom_y = line_top
                break
        else:
            bottom_y = self.page['height']

        self.page_row_pos = [y for y in self.pos_y if top_y <= y <= bottom_y]
        print("> page %d: %d lines between [%f, %f]" % (self.page_no, len(self.page_row_pos), top_y, bottom_y))


    def to_grid(self):
        grid = make_grid_from_positions(self.page_col_pos, self.page_row_pos)
        n_rows = len(grid)
        n_cols = len(grid[0])
        print("> page %d: grid with %d rows, %d columns" % (self.page_no, n_rows, n_cols))

        output_files_basename = self.file[:self.file.rindex('.')]
        page_grids_file = os.path.join(self.OUTPUT_PATH, output_files_basename + '.pagegrids_p%s_only.json' % self.page_no)
        print("saving page grids JSON file to '%s'" % page_grids_file)
        save_page_grids({self.page_no: grid}, page_grids_file)

        datatable = fit_texts_into_grid(self.page['texts'], grid)
        df = datatable_to_dataframe(datatable)
        df.head(10)

        df.to_csv('output.csv', index=False)
        df.to_excel('output.xlsx', index=False)
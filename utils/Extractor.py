import os
import ntpath

from PyQt5.QtWidgets import QTextEdit

from pdftabextract.common import read_xml, parse_pages

from utils import helpers
from utils.PageExtractor import PageExtractor

class Extractor:
    def __init__(self, file):
        self.tedit = helpers.getMainWindow().findChild(QTextEdit)
        self.file = ntpath.basename(file)
        self.filepath_rel = ntpath.relpath(file)
        self.OUTPUT_PATH = os.path.join(ntpath.dirname(self.filepath_rel),'generated_output')
        print(self.OUTPUT_PATH)
        if not os.path.exists(self.OUTPUT_PATH) :
            os.mkdir(self.OUTPUT_PATH)
            print('directory created : ',self.OUTPUT_PATH)


    def parseXML(self):
        # Load the XML that was generated with pdftohtml
        self.xmltree, self.xmlroot = read_xml(self.filepath_rel)
        # parse it and generate a dict of pages
        pages = parse_pages(self.xmlroot)
        self.tedit.append('parssing XML file...')
        p_num = 3
        page = pages[p_num]
        # self.save_page_as_image(page, p_num)
        page = PageExtractor(page, p_num, self.file, self.OUTPUT_PATH, self.xmltree)
        page.save_page_as_image()
        page.find_skew_rotation()
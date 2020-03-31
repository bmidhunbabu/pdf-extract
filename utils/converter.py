import os
import sys
import ntpath

from utils import helpers

def toXML(filepath):
    filename = ntpath.relpath(filepath)
    outfile = filename + '.xml'
    result = helpers.run_shell_cmd('pdftohtml -c -hidden -xml %s %s' % (filename, outfile))
    if result:
        return outfile
    else:
        return False
import typing
import sys
import os

from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.QtCore import QDir

def getMainWindow() -> typing.Union[QDialog, None]:
    # Global function to find the (open) QMainWindow in application
    app = QApplication.instance()
    for widget in app.topLevelWidgets():
        if isinstance(widget, QDialog):
            return widget
    return None

def printerr(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def run_shell_cmd(cmd):
    print('> running shell command: %s' % cmd)
    ret = os.system(cmd)
    if ret != 0:
        printerr('shell command failed with error code %d' % ret)
        return False
    else:
        return True


def openFileNameDialog():
    print("opening file dialog")
    file = None
    try :
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setFilter(QDir.Dirs | QDir.Files | QDir.Drives)
        dialog.setNameFilter("PDF files (*.pdf)")
        if dialog.exec_():
            filenames = dialog.selectedFiles()
            file = filenames[0]
            print("Selected file : " + file)
        else:
            print("no file selected")
    except Exception as e:
        print(str(e))
    return file
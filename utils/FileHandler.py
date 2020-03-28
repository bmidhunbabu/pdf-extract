from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QDir

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
# MyApp.py
# D. Thiebaut
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from UIComponents.UiDialog import Ui_Dialog
import sys
from utils import FileHandler


class MainWindow(Ui_Dialog):
    def __init__(self):
        '''Initialize the super class
        '''
        super().__init__()

    def setupUi(self, MW):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi(MW)

    def OpePdfFile(self):
        try:
            filepath = FileHandler.openFileNameDialog()
        except Exception as e:
            print(str(e))
        if (filepath != None):
            self.lineEdit.insert(filepath)
            self.log("Selected file " + filepath)
            self.convertButton.setEnabled(True)

    def convertFile(self):
        self.log("Working on it...")
        self.log("Detecting table layouts...")


    def exportFile(self):
        pass


    def log(self, messsage):
        self.textEdit.append(messsage)
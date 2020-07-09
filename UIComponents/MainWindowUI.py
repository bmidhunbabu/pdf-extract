# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow

class Ui_MainWindow(QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(464, 204)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 309, 13))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 16))
        self.label_2.setObjectName("label_2")
        self.fileName = QtWidgets.QLineEdit(self.centralwidget)
        self.fileName.setEnabled(False)
        self.fileName.setGeometry(QtCore.QRect(10, 30, 309, 23))
        self.fileName.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.fileName.setObjectName("fileName")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 60, 309, 16))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 16))
        self.label_3.setObjectName("label_3")
        self.opePdfFileButton = QtWidgets.QPushButton(self.centralwidget)
        self.opePdfFileButton.setGeometry(QtCore.QRect(330, 30, 119, 23))
        self.opePdfFileButton.setMinimumSize(QtCore.QSize(100, 0))
        self.opePdfFileButton.setObjectName("opePdfFileButton")
        # self.doublePages = QtWidgets.QCheckBox(self.centralwidget)
        # self.doublePages.setGeometry(QtCore.QRect(10, 80, 131, 17))
        # self.doublePages.setObjectName("doublePages")
        self.preprocess = QtWidgets.QCheckBox(self.centralwidget)
        self.preprocess.setGeometry(QtCore.QRect(10, 100, 141, 17))
        self.preprocess.setObjectName("preprocess")
        self.verticalLines = QtWidgets.QCheckBox(self.centralwidget)
        self.verticalLines.setGeometry(QtCore.QRect(10, 120, 201, 17))
        self.verticalLines.setObjectName("verticalLines")
        self.horizontalLines = QtWidgets.QCheckBox(self.centralwidget)
        self.horizontalLines.setGeometry(QtCore.QRect(10, 140, 191, 17))
        self.horizontalLines.setObjectName("horizontalLines")
        self.convertButton = QtWidgets.QPushButton(self.centralwidget)
        self.convertButton.setGeometry(QtCore.QRect(330, 110, 121, 23))
        self.convertButton.setStyleSheet("")
        self.convertButton.setObjectName("convertButton")
        self.exportButton = QtWidgets.QPushButton(self.centralwidget)
        self.exportButton.setGeometry(QtCore.QRect(330, 140, 121, 23))
        self.exportButton.setStyleSheet("")
        self.exportButton.setObjectName("exportButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 60, 47, 13))
        self.label.setObjectName("label")
        self.pageNumber = QtWidgets.QSpinBox(self.centralwidget)
        self.pageNumber.setGeometry(QtCore.QRect(330, 80, 121, 22))
        self.pageNumber.setMinimum(1)
        self.pageNumber.setObjectName("pageNumber")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 464, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Input File"))
        self.label_3.setText(_translate("MainWindow", "Options"))
        self.opePdfFileButton.setText(_translate("MainWindow", "Open PDF File"))
        # self.doublePages.setText(_translate("MainWindow", "Two Pages on a sheet"))
        self.preprocess.setText(_translate("MainWindow", "Preprocess Document"))
        self.verticalLines.setText(_translate("MainWindow", "No Vertical Lines"))
        self.horizontalLines.setText(_translate("MainWindow", "No Horizontal Lines"))
        self.convertButton.setText(_translate("MainWindow", "Convert"))
        self.exportButton.setText(_translate("MainWindow", "Export"))
        self.label.setText(_translate("MainWindow", "Page No."))




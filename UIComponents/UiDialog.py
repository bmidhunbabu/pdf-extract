# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(470, 234)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(330, 10, 121, 181))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.OpePdfFileButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.OpePdfFileButton.setMinimumSize(QtCore.QSize(100, 0))
        self.OpePdfFileButton.setObjectName("OpePdfFileButton")
        self.verticalLayout_2.addWidget(self.OpePdfFileButton)
        self.checkBox = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout_2.addWidget(self.checkBox)
        self.convertButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.convertButton.setStyleSheet("")
        self.convertButton.setObjectName("convertButton")
        self.verticalLayout_2.addWidget(self.convertButton)
        spacerItem = QtWidgets.QSpacerItem(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setMaximumSize(QtCore.QSize(16777215, 16))
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox.setStyleSheet("")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox)
        self.exportButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.exportButton.setStyleSheet("")
        self.exportButton.setObjectName("exportButton")
        self.verticalLayout_2.addWidget(self.exportButton)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 311, 181))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.horizontalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout.addWidget(self.lineEdit)
        self.textEdit = QtWidgets.QTextEdit(self.horizontalLayoutWidget)
        # self.textEdit.setEnabled(False)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 470, 21))
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
        self.OpePdfFileButton.setText(_translate("MainWindow", "Open PDF File"))
        self.checkBox.setText(_translate("MainWindow", "Double Pages"))
        self.convertButton.setText(_translate("MainWindow", "Convert"))
        self.label.setText(_translate("MainWindow", "Export Format"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Excel (.xslt)"))
        self.comboBox.setItemText(1, _translate("MainWindow", "CSV (.csv)"))
        self.comboBox.setItemText(2, _translate("MainWindow", "JSON (.json)"))
        self.exportButton.setText(_translate("MainWindow", "Export"))


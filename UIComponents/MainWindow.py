from UIComponents.MainWindowUI import Ui_MainWindow
from utils import helpers
from Extractor.PDFExtractor import Extractor


class MainWindow(Ui_MainWindow):

    _file = None
    _extractor = None
    _tabular_data = None
    _page_number = None

    def __init__(self):
        super().__init__()

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.connectSlots()

    def connectSlots(self):
        self.convertButton.setDisabled(True)
        self.exportButton.setDisabled(True)
        self.opePdfFileButton.clicked.connect(self.openPdfFile)
        self.convertButton.clicked.connect(self.convertFile)
        self.exportButton.clicked.connect(self.exportFile)

    def _log(self, message):
        self.statusbar.showMessage(message)

    def setFileName(self,filename):
        self.fileName.setText(filename)
        self._log("Selected file " + filename)

    def getFileName(self):
        return self._file

    def openPdfFile(self):
        try:
            self._file = helpers.openFileNameDialog()
            self.setFileName(self._file)
            self.convertButton.setEnabled(True)
        except Exception as e:
            print(str(e))

    def convertFile(self):
        if self._file is None:
            self._log("No File Chosen")
            return
        self._extractor = Extractor(self._file)
        self._extractor.setStatusBar(self.statusbar)
        if self.preprocess.isChecked():
            self._extractor.setPreProcess(True)
        if self.verticalLines.isChecked():
            self._extractor.setVerticalLines(False)
        if self.horizontalLines.isChecked():
            self._extractor.setHorizontalLines(False)
        # if self.doublePages.isChecked():
        #     self._extractor.setDoublePages(True)
        self._log("Extracting pages from file")
        self._extractor.convertPages()
        self._page_number = self.pageNumber.value()
        self._tabular_data = self._extractor.parsePage(self._page_number)
        self.exportButton.setEnabled(True)

    def exportFile(self):
        self._log("Exporting detected data")
        if self._tabular_data is None:
            self._log("No data found")
            return False
        excel = self._file[:self._file.rindex('.')] + ".xlsx"
        sheet = "Sheet1"
        if self._page_number is not None:
            sheet = "Page %s" % self._page_number
        self._tabular_data.to_excel(excel, sheet_name=sheet, encoding="utf-8")
        self._log("File saved to %s" % excel)
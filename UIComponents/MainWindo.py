import ntpath
import time

from UIComponents.UiDialog import Ui_Dialog
from utils import converter
from utils import helpers
from utils.Extractor import Extractor


class MainWindow(Ui_Dialog):
    # Initialize the super class
    def __init__(self):
        super().__init__()


    # Setup the UI of the super class, and add here code
    # that relates to the way we want our UI to operate.
    def setupUi(self, MW):
        super().setupUi(MW)
        # connecting the slots
        self.OpePdfFileButton.clicked.connect(self.OpePdfFile)
        self.convertButton.clicked.connect(self.convertFile)
        self.exportButton.clicked.connect(self.exportFile)


    def OpePdfFile(self):
        try:
            filepath = helpers.openFileNameDialog()
            if (filepath != None):
                self.lineEdit.insert(filepath)
                self.log("Selected file " + ntpath.basename(filepath))
                self.convertButton.setEnabled(True)
        except Exception as e:
            print(str(e))


    def convertFile(self):
        try:
            self.log('Converting to XML...')
            self.convertButton.setEnabled(False)
            filepath = self.lineEdit.text()
            time.sleep(2.4)
            xml_file = converter.toXML(filepath)
            if xml_file == False:
                self.log('Could not convert to XML')
                return False
            self.log('Finished converting to XML')
            self.log('XML saved to ' + ntpath.abspath(xml_file))
            extractor = Extractor(xml_file)
            extractor.parseXML()
            self.convertButton.setEnabled(True)
        except Exception as e:
            print(str(e))

    def exportFile(self):
        pass


    def log(self, messsage):
        self.textEdit.append(messsage)
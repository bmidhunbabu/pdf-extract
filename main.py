from UIComponents.MainWindo import MainWindow
from PyQt5 import QtWidgets

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Window = QtWidgets.QMainWindow()
    ui = MainWindow()
    ui.setupUi(Window)
    Window.show()
    sys.exit(app.exec_())
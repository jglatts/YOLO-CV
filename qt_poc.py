import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QMessageBox, QVBoxLayout, QGroupBox, QHBoxLayout
)
from PyQt6.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt6 Example")
        self.setFixedSize(800, 600)  # fixed size, not resizable

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)  # padding
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Group box
        group_box = QGroupBox("Controls")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        # Buttonsinside group box
        self.button = QPushButton("Click Me")
        self.button.clicked.connect(self.testButton_Callback)
        self.button_cam = QPushButton("Click Me")
        self.button_cam.clicked.connect(self.openCam_Callback)
        group_layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignLeft)
        group_layout.addWidget(self.button_cam, alignment=Qt.AlignmentFlag.AlignLeft)

        # Add group box to main layout
        main_layout.addWidget(group_box)

    def testButton_Callback(self):
        self.show_popup("button clicked!", "JDG")

    def openCam_Callback(self):
        self.show_popup("opening cam!", "JDG")

    def show_popup(self, msg="", title=""):
        msgBox = QMessageBox()
        msgBox.setWindowTitle(msg)
        msgBox.setText(title)
        msgBox.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

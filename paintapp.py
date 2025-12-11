from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paint with PyQt5")
        self.setGeometry(100, 100, 800, 600)
        
        # Set Window Icon
        self.setWindowIcon(QIcon("icons/app_icon.png"))  # Change to your icon path

        # Creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        # Variables
        self.drawing = False
        self.brushSize = 4
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
        self.fillMode = False  # Track fill tool usage

        # Creating menu bar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        b_size = mainMenu.addMenu("Brush Size")
        b_color = mainMenu.addMenu("Brush Color")
        editMenu = mainMenu.addMenu("Edit")

        # Save action
        saveAction = QAction(QIcon("icons/save.png"), "Save", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.save)
        fileMenu.addAction(saveAction)

        # Clear action
        clearAction = QAction(QIcon("icons/clear.png"), "Clear", self)
        clearAction.setShortcut("Ctrl+C")
        clearAction.triggered.connect(self.clear)
        fileMenu.addAction(clearAction)

        # Exit action
        exitAction = QAction(QIcon("icons/exit.png"), "Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.triggered.connect(self.close)  # Close the application
        fileMenu.addAction(exitAction)

        # Brush size actions
        for size in [4, 7, 9, 12]:
            action = QAction(f"{size}px", self)
            action.triggered.connect(lambda checked, s=size: self.setBrushSize(s))
            b_size.addAction(action)

        # Color picker action
        colorPicker = QAction(QIcon("icons/color_picker.png"), "Choose Color", self)
        colorPicker.triggered.connect(self.selectColor)
        b_color.addAction(colorPicker)

        # Eraser action
        eraserAction = QAction(QIcon("icons/eraser.png"), "Eraser", self)
        eraserAction.triggered.connect(self.useEraser)
        editMenu.addAction(eraserAction)

        # Fill color action
        fillAction = QAction(QIcon("icons/fill.png"), "Fill Color", self)
        fillAction.triggered.connect(self.enableFillMode)
        editMenu.addAction(fillAction)

    # Mouse events
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.fillMode:  # If fill mode is active, apply flood fill
                self.floodFill(event.pos(), self.brushColor)
                self.fillMode = False  # Reset fill mode after use
            else:
                self.drawing = True
                self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    # Paint event
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    # Save method
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)")
        if filePath:
            self.image.save(filePath)

    # Clear method
    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    # Set brush size
    def setBrushSize(self, size):
        self.brushSize = size

    # Select color using color picker
    def selectColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.brushColor = color

    # Eraser function
    def useEraser(self):
        self.brushColor = Qt.white

    # Enable fill mode
    def enableFillMode(self):
        self.fillMode = True

    # Flood Fill function (Fixed for Qt.GlobalColor)
    def floodFill(self, startPos, newColor):
        image = self.image
        width, height = image.width(), image.height()

        targetColor = image.pixel(startPos)
        newRgbColor = QColor(newColor).rgb()  # Convert GlobalColor to QColor

        if targetColor == newRgbColor:
            return  # No need to fill if the color is already the same

        # Stack-based flood fill (avoids recursion limit issues)
        stack = [startPos]

        while stack:
            point = stack.pop()
            x, y = point.x(), point.y()

            if x < 0 or x >= width or y < 0 or y >= height:
                continue  # Skip out-of-bounds pixels

            if image.pixel(QPoint(x, y)) != targetColor:
                continue  # Skip already filled or different colors

            image.setPixel(QPoint(x, y), newRgbColor)

            # Add neighboring pixels to stack
            stack.append(QPoint(x + 1, y))
            stack.append(QPoint(x - 1, y))
            stack.append(QPoint(x, y + 1))
            stack.append(QPoint(x, y - 1))

        self.update()

    # Close event method to ensure the app quits
    def closeEvent(self, event):
        print("Closing application...")
        event.accept()  # Accept the close event and quit properly

# Create app
App = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(App.exec())

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from math import sqrt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMenu
import sys

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paint with PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # Creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        # Variables
        self.drawing = False
        self.brushSize = 4
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
        self.fillMode = False  # Track fill tool usage
        self.currentShape = None  # No shape selected initially
        self.points = []  # Store points for shape recognition
        self.shapeRecognitionMode = False  # Track whether shape recognition is enabled

        # Creating menu bar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        b_size = mainMenu.addMenu("Brush Size")
        b_color = mainMenu.addMenu("Brush Color")
        editMenu = mainMenu.addMenu("Edit")
        shapeMenu = mainMenu.addMenu("Shapes")

        # Shape menu actions
        shapes = ["Circle", "Square", "Rectangle", "Triangle", "Star"]
        for shape in shapes:
            action = QAction(shape, self)
            action.triggered.connect(lambda checked, s=shape: self.setShape(s))
            shapeMenu.addAction(action)

        # Brush size actions
        for size in [2, 4, 7, 9, 12]:
            action = QAction(f"{size}px", self)
            action.triggered.connect(lambda checked, s=size: self.setBrushSize(s))
            b_size.addAction(action)

        # Color picker action
        colorPicker = QAction("Choose Color", self)
        colorPicker.triggered.connect(self.selectColor)
        b_color.addAction(colorPicker)

        # Eraser action
        eraserAction = QAction("Eraser", self)
        eraserAction.setIcon(QIcon("icons/eraser.png"))  # Set icon for the eraser
        eraserAction.triggered.connect(self.useEraser)
        editMenu.addAction(eraserAction)

        # Eraser size menu
        e_size = mainMenu.addMenu("Eraser Size")
        for size in [2, 4, 7, 9, 12]:
            action = QAction(f"{size}px", self)
            action.triggered.connect(lambda checked, s=size: self.setEraserSize(s))
            e_size.addAction(action)

        # Fill color action
        fillAction = QAction("Fill Color", self)
        fillAction.setIcon(QIcon("icons/color-wheel.png"))
        fillAction.triggered.connect(self.enableFillMode)
        editMenu.addAction(fillAction)

        # Smart shape
        shapeRecognitionAction = QAction("Smart Shape Recognition", self)
        shapeRecognitionAction.triggered.connect(self.toggleShapeRecognition)
        editMenu.addAction(shapeRecognitionAction)

        # stroke style
        b_style = self.menuBar().addMenu("Brush Style")

        # Define stroke styles
        styles = {
            "Solid": Qt.SolidLine,
            "Dashed": Qt.DashLine,
            "Dotted": Qt.DotLine,
            "Dash-Dot": Qt.DashDotLine,
            "Dash-Dot-Dot": Qt.DashDotDotLine
        }

        self.brushStyle = Qt.SolidLine  # Default style

        # Create actions for each style
        for name, style in styles.items():
            action = QAction(name, self)
            action.triggered.connect(lambda checked, s=style: self.setBrushStyle(s))
            b_style.addAction(action)

    def setShape(self, shape):
        self.currentShape = shape

    def toggleShapeRecognition(self):
        self.shapeRecognitionMode = not self.shapeRecognitionMode

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.currentShape:
                self.drawing = True
                self.lastPoint = event.pos()
                self.points = [event.pos()]  # Start recording points for shape
            else:
                self.drawing = True
                self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            if self.currentShape:
                self.drawShape(event.pos())
            else:
                painter = QPainter(self.image)
                pen = QPen(self.brushColor, self.brushSize, self.brushStyle, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.lastPoint, event.pos())
                self.lastPoint = event.pos()
                self.update()
            self.points.append(event.pos())  # Store points for shape recognition

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            if self.shapeRecognitionMode and self.points:  # Only recognize shapes if enabled
                self.recognizeShape()

    def drawShape(self, pos):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        shapeSize = 50

        if self.currentShape == "Circle":
            radius = int(sqrt((pos.x() - self.lastPoint.x())**2 + (pos.y() - self.lastPoint.y())**2))
            painter.drawEllipse(pos, shapeSize, shapeSize)
        elif self.currentShape == "Square":
            size = max(abs(pos.x() - self.lastPoint.x()), abs(pos.y() - self.lastPoint.y()))
            painter.drawRect(self.lastPoint.x(), self.lastPoint.y(), size, size)
        elif self.currentShape == "Rectangle":
            width = abs(pos.x() - self.lastPoint.x())
            height = abs(pos.y() - self.lastPoint.y())
            painter.drawRect(self.lastPoint.x(), self.lastPoint.y(), width, height)
        elif self.currentShape == "Triangle":
            points = [self.lastPoint, QPoint(pos.x(), self.lastPoint.y()), QPoint(pos.x(), pos.y())]
            painter.drawPolygon(QPolygon(points))
        elif self.currentShape == "Star":
            self.drawStar(painter, self.lastPoint, pos)

        self.update()

    def drawStar(self, painter, start, end):
        points = [
            QPoint(start.x(), start.y() - 40),
            QPoint(start.x() + 12, start.y() - 12),
            QPoint(start.x() + 40, start.y() - 12),
            QPoint(start.x() + 18, start.y() + 6),
            QPoint(start.x() + 28, start.y() + 34),
            QPoint(start.x(), start.y() + 18),
            QPoint(start.x() - 28, start.y() + 34),
            QPoint(start.x() - 18, start.y() + 6),
            QPoint(start.x() - 40, start.y() - 12),
            QPoint(start.x() - 12, start.y() - 12)
        ]
        painter.drawPolygon(QPolygon(points))

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)")
        if filePath:
            self.image.save(filePath)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def setBrushSize(self, size):
        self.brushSize = size

    def setBrushStyle(self, style):
        self.brushStyle = style

    def selectColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.brushColor = color

    def useEraser(self):
        self.brushColor = Qt.white

    def setEraserSize(self, size):
        self.brushSize = size

    def enableFillMode(self):
        self.fillMode = True

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

    
        


# Create app
App = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(App.exec())

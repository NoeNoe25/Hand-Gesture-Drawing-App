from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from math import sqrt

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
        self.eraserSize = 4  # Separate eraser size
        self.lastBrushColor = Qt.black  # Default brush color


        # Creating menu bar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        b_size = mainMenu.addMenu("Brush Size")
        b_color = mainMenu.addMenu("Brush Color")
        editMenu = mainMenu.addMenu("Edit")

        # Save action
        saveAction = QAction("Save", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.save)
        fileMenu.addAction(saveAction)

        # Clear action
        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Ctrl+C")
        clearAction.triggered.connect(self.clear)
        fileMenu.addAction(clearAction)


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
        fillAction.triggered.connect(self.enableFillMode)
        editMenu.addAction(fillAction)

        # Smart shape
        self.points = []  # Store points for shape recognition
        
        self.shapeRecognitionMode = False  # Track whether shape recognition is enabled
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

    # smart shape
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.fillMode:  # If fill mode is active, apply flood fill
                self.floodFill(event.pos(), self.brushColor)
                #self.fillMode = False  # Reset fill mode after use
            else:
                self.drawing = True
                self.lastPoint = event.pos()
                self.points = [event.pos()]  # Start recording points

        if self.shapeRecognitionMode:
            # Save a backup of the image before sketching
            self.backupImage = self.image.copy()

    def toggleShapeRecognition(self):
        self.shapeRecognitionMode = not self.shapeRecognitionMode

    # smart shape
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            self.points.append(event.pos())  # Store points for analysis
            painter = QPainter(self.image)
            pen = QPen(self.brushColor, self.brushSize, self.brushStyle, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    # smart shape
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

            if self.shapeRecognitionMode and self.points:  # Only recognize shapes if enabled
                self.recognizeShape()

    # Paint event
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())


    # Save method
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*)")
        if filePath:
            self.image.save(filePath)


    def clear(self):
        self.image.fill(Qt.white)
        self.update()


    # Set brush size
    def setBrushSize(self, size):
        self.brushSize = size

    # Select brush style
    def setBrushStyle(self, style):
        self.brushStyle = style
        if self.brushColor == Qt.white:  # If currently using the eraser
            self.brushColor = self.lastBrushColor  # Restore last brush color

        
    # Select color using color picker
    def selectColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.brushColor = color
            self.lastBrushColor = color  # Save the last chosen brush color


    def useEraser(self):
        self.lastBrushColor = self.brushColor  # Save the current brush color before erasing
        self.brushColor = Qt.white
        self.brushStyle = Qt.SolidLine  # Reset brush style to solid
        self.fillMode = False  # Ensure normal eraser functionality
        self.shapeRecognitionMode = False  # Turn off smart shape recognition
        self.brushSize = self.eraserSize  # Apply the correct eraser size

    
    # Eraser size
    def setEraserSize(self, size):
        self.eraserSize = size  # Only affects the eraser
        if self.brushColor == Qt.white:  # If currently using the eraser
            self.brushSize = self.eraserSize  # Apply eraser size


    # Enable fill mode
    def enableFillMode(self):
        self.fillMode = True


    # smart shape
    def recognizeShape(self):
        if len(self.points) < 5:  # Not enough points to analyze
            return

        # Restore the original image to erase the sketch
        self.image = self.backupImage.copy()

        # Get bounding box
        min_x = min(p.x() for p in self.points)
        max_x = max(p.x() for p in self.points)
        min_y = min(p.y() for p in self.points)
        max_y = max(p.y() for p in self.points)

        width = max_x - min_x
        height = max_y - min_y

        # Calculate distances between first and last points
        start = self.points[0]
        end = self.points[-1]
        dist = sqrt((start.x() - end.x())**2 + (start.y() - end.y())**2)

        # Decide shape
        if dist < max(width, height) * 0.2:  # Almost closed shape
            shape = "circle"
        elif width < height * 2.2 and height < width * 1.2:  # Roughly square
            shape = "rectangle"
        # elif len(self.points) >= 10:  # If we have enough points
        #     shape = "triangle"
        else:
            shape = "line"

        self.drawRecognizedShape(shape, min_x, min_y, max_x, max_y)

    def drawRecognizedShape(self, shape, x1, y1, x2, y2):
        painter = QPainter(self.image)
        pen = QPen(self.brushColor, self.brushSize, self.brushStyle, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)

        if shape == "line":
            painter.drawLine(x1, y1, x2, y2)
        elif shape == "rectangle":
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        elif shape == "circle":
            diameter = max(x2 - x1, y2 - y1)
            painter.drawEllipse(x1, y1, diameter, diameter)
        
        self.update()


    # Flood Fill function (Fixed for Qt.GlobalColor)
    def floodFill(self, startPos, newColor):
        image = self.image
        width, height = image.width(), image.height()

        targetColor = QColor(image.pixel(startPos))  # Convert pixel color to QColor
        newColor = QColor(newColor)  # Ensure newColor is a QColor

        if targetColor == newColor:
            return  # No need to fill if the color is already the same

        stack = [startPos]
        visited = set()  # To avoid redundant checks

        while stack:
            point = stack.pop()
            x, y = point.x(), point.y()

            if (x, y) in visited:
                continue
            visited.add((x, y))

            if x < 0 or x >= width or y < 0 or y >= height:
                continue  # Skip out-of-bounds pixels

            if QColor(image.pixel(x, y)) != targetColor:
                continue  # Skip already filled or different colors

            image.setPixelColor(x, y, newColor)  # Fill pixel with new color

            # Add neighboring pixels to stack
            stack.append(QPoint(x + 1, y))
            stack.append(QPoint(x - 1, y))
            stack.append(QPoint(x, y + 1))
            stack.append(QPoint(x, y - 1))

        self.update()  # Refresh the canvas


# Create app
App = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(App.exec())

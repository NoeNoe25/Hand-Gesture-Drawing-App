from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paint with PyQt5")
        self.setGeometry(100, 100, 800, 600)

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 4
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
        self.currentShape = None  # Track selected shape

        mainMenu = self.menuBar()
        shapeMenu = mainMenu.addMenu("Shapes")

        shapes = ["Circle", "Square", "Rectangle", "Triangle", "Star"]
        for shape in shapes:
            action = QAction(shape, self)
            action.triggered.connect(lambda checked, s=shape: self.setShape(s))
            shapeMenu.addAction(action)

    def setShape(self, shape):
        self.currentShape = shape

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.currentShape:
            self.drawShape(event.pos())

    def drawShape(self, pos):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        if self.currentShape == "Circle":
            painter.drawEllipse(pos, 50, 50)
        elif self.currentShape == "Square":
            painter.drawRect(pos.x() - 25, pos.y() - 25, 50, 50)
        elif self.currentShape == "Rectangle":
            painter.drawRect(pos.x() - 40, pos.y() - 25, 80, 50)
        elif self.currentShape == "Triangle":
            points = [QPoint(pos.x(), pos.y() - 40), QPoint(pos.x() - 30, pos.y() + 20), QPoint(pos.x() + 30, pos.y() + 20)]
            painter.drawPolygon(QPolygon(points))
        elif self.currentShape == "Star":
            self.drawStar(painter, pos)
        
        self.update()

    def drawStar(self, painter, pos):
        points = [
            QPoint(pos.x(), pos.y() - 40),
            QPoint(pos.x() + 12, pos.y() - 12),
            QPoint(pos.x() + 40, pos.y() - 12),
            QPoint(pos.x() + 18, pos.y() + 6),
            QPoint(pos.x() + 28, pos.y() + 34),
            QPoint(pos.x(), pos.y() + 18),
            QPoint(pos.x() - 28, pos.y() + 34),
            QPoint(pos.x() - 18, pos.y() + 6),
            QPoint(pos.x() - 40, pos.y() - 12),
            QPoint(pos.x() - 12, pos.y() - 12)
        ]
        painter.drawPolygon(QPolygon(points))

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

App = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(App.exec())

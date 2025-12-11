import sys
import torch
from torch import nn
from torchvision import transforms, models
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import numpy as np

class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Style Transfer")
        self.setGeometry(100, 100, 600, 600)
        
        self.layout = QVBoxLayout()
        
        # UI Elements
        self.loadButton = QPushButton("Load Image")
        self.applyButton = QPushButton("Apply Style Transfer")
        self.imageLabel = QLabel()
        
        self.layout.addWidget(self.loadButton)
        self.layout.addWidget(self.applyButton)
        self.layout.addWidget(self.imageLabel)
        
        self.setLayout(self.layout)
        
        # Connect buttons to functions
        self.loadButton.clicked.connect(self.load_image)
        self.applyButton.clicked.connect(self.apply_style_transfer)
        
        # Initialize variables
        self.input_image = None
        self.output_image = None
        
        # Pretrained VGG model for style transfer
        self.vgg_model = models.vgg19(pretrained=True).features.eval()
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        
    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file:
            self.input_image = Image.open(file).convert('RGB')
            self.display_image(self.input_image)
        
    def apply_style_transfer(self):
        if self.input_image:
            # Load and transform input image
            input_tensor = self.transform(self.input_image).unsqueeze(0).cuda()

            # Random noise image as the target for style transfer
            target_tensor = input_tensor.clone().requires_grad_(True)

            # Style loss and content loss
            content_weight = 1e4
            style_weight = 1e2

            # Optimizer
            optimizer = torch.optim.LBFGS([target_tensor])

            def closure():
                target_tensor.data.clamp_(0, 255)
                optimizer.zero_grad()

                # Extract features of input and target image using VGG model
                content_loss = self.compute_content_loss(input_tensor, target_tensor)
                style_loss = self.compute_style_loss(input_tensor, target_tensor)

                # Calculate total loss
                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()

                return total_loss

            # Run the optimization
            optimizer.step(closure)

            # Convert tensor back to image
            self.output_image = target_tensor.detach().cpu().squeeze(0)
            self.output_image = self.convert_tensor_to_image(self.output_image)
            self.display_image(self.output_image)

    def compute_content_loss(self, input_tensor, target_tensor):
        input_features = self.extract_features(input_tensor)
        target_features = self.extract_features(target_tensor)
        content_loss = nn.MSELoss()(input_features, target_features)
        return content_loss

    def compute_style_loss(self, input_tensor, target_tensor):
        input_features = self.extract_features(input_tensor)
        target_features = self.extract_features(target_tensor)
        
        style_loss = 0.0
        for input_f, target_f in zip(input_features, target_features):
            input_gram = self.gram_matrix(input_f)
            target_gram = self.gram_matrix(target_f)
            style_loss += nn.MSELoss()(input_gram, target_gram)
        
        return style_loss

    def extract_features(self, x):
        layers = []
        for name, layer in self.vgg_model._modules.items():
            x = layer(x)
            if name == '4':  # Content layer (conv_4_2)
                layers.append(x)
            if name == '21':  # Style layers (conv_5_2)
                layers.append(x)
        return layers

    def gram_matrix(self, x):
        _, c, h, w = x.size()
        features = x.view(c, h * w)
        gram = torch.mm(features, features.t())
        return gram / (c * h * w)

    def convert_tensor_to_image(self, tensor):
        image = tensor.to('cpu').clone().detach().numpy()
        image = image.squeeze(0).transpose(1, 2, 0)
        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

    def display_image(self, image):
        image = image.convert("RGB")
        data = image.tobytes()
        qim = QImage(data, image.width, image.height, image.width * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qim)
        self.imageLabel.setPixmap(pix.scaled(self.imageLabel.size(), aspectRatioMode=True))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.show()
    sys.exit(app.exec_())

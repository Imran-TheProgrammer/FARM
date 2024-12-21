# agents/disease_agent.py
from .base import Agent
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
import io

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_accuracy': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accuracy = [x['val_accuracy'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'val_loss': epoch_loss, 'val_accuracy': epoch_accuracy}

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                    nn.Flatten(),
                                    nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

class DiseaseDetectionAgent(Agent):
    def __init__(self):
        super().__init__("DiseaseDetectionAgent", ["disease", "plant disease", "detect disease", "analyze plant"])
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.input_shape = (256, 256)
        self.classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
            'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 
            'Cherry___healthy', 'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 
            'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 
            'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
            'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
            'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
            'Tomato___healthy'
        ]

    def _load_model(self):
        """Load the trained PyTorch model."""
        try:
            model = ResNet9(3, 38)  # 3 channels, 38 classes
            model.load_state_dict(torch.load('plant-disease-model.pth', 
                                           map_location=self.device))
            model = model.to(self.device)
            model.eval()
            self.logger.info("Model loaded successfully")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image_data):
        """Preprocess image for model input."""
        try:
            # Open image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_data)
                
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Define transforms
            transform = transforms.Compose([
                transforms.Resize(self.input_shape),
                transforms.ToTensor(),
            ])
            
            # Transform the image
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def get_disease_info(self, disease_class):
        """Get information about the detected disease."""
        try:
            with open('disease_info.json', 'r') as f:
                disease_info = json.load(f)
            return disease_info.get(disease_class, {
                'name': disease_class,
                'description': 'No detailed information available.',
                'treatments': ['Consult a local agricultural expert.']
            })
        except Exception as e:
            self.logger.error(f"Error loading disease info: {str(e)}")
            return {
                'name': disease_class,
                'description': 'Information unavailable.',
                'treatments': ['Consult a local agricultural expert.']
            }

    def execute(self, image_data: bytes) -> Dict[str, Any]:
        """Execute disease detection on an image."""
        try:
            print("Starting disease detection...")  # Debug log
            
            # Preprocess image
            processed_image = self.preprocess_image(image_data)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(processed_image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = self.classes[predicted.item()]
                confidence = confidence.item()
            
            print(f"Predicted class: {predicted_class}, Confidence: {confidence}")  # Debug log
            
            # Get disease information
            disease_info = self.get_disease_info(predicted_class)
            
            # Format the response text
            response = (
                f"**Disease Analysis Results**\n\n"
                f"**Disease Detected:** {disease_info['name']}\n"
                f"**Confidence:** {confidence:.2%}\n\n"
                f"**Description:**\n{disease_info['description']}\n\n"
                f"**Recommended Treatments:**\n"
            )
            
            # Add treatments as numbered list
            for i, treatment in enumerate(disease_info['treatments'], 1):
                response += f"{i}. {treatment}\n"

            print(f"Generated response: {response}")  # Debug log
            return response
            
        except Exception as e:
            print(f"Error in disease detection: {str(e)}")  # Debug log
            return "I apologize, but I encountered an error while analyzing the image. Please ensure you've uploaded a clear image of a plant and try again."
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Base class for the model
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
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

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

def predict_plant_disease(image_path, model_path):
    """
    Predict plant disease from an image file
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the saved model file
    
    Returns:
        str: Predicted disease class
    """
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture before loading state
    model = ResNet9(3, 38)  # 3 channels, 38 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Define the class names (in the same order as training)
    classes = [
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
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Define the same transforms used during training
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Transform the image
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]
        
        # Display the image and prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Predicted Disease: {predicted_class.replace("___", " - ")}')
        plt.show()
        
        return predicted_class
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual paths
    image_path = "Test Images/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.jpg"
    model_path = "plant-disease-model.pth"
    
    prediction = predict_plant_disease(image_path, model_path)
    if prediction:
        print(f"\nPredicted class: {prediction}")
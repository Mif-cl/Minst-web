import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from ImageClassifier import ImageClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load complete model
print("Loading complete model...")
model = ImageClassifier().to(device)
model.load_state_dict(torch.load('checkpoint.pt', map_location=device))
model.eval()

def predict_image(model, image_path, device):
    """
    Predict the class of a single image
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities.squeeze().tolist()
    
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, 0, 0
    
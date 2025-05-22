from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load trained model
from model import LeNet 
model = LeNet()
model.load_state_dict(torch.load("mnist_trained_model.pth", map_location=torch.device("cpu")))
model.eval()

# Initialize API
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")  # Convert uploaded file to grayscale
    image = transform(image).unsqueeze(0)  # Apply transformation

    # Perform inference
    with torch.no_grad():
        output = model(image)
        predicted_label = output.argmax(dim=1).item()

    return {"predicted_digit": predicted_label}

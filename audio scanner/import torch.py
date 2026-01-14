import torch
import torch.nn as nn
import librosa
import numpy as np
from torchvision import models, transforms
from PIL import Image

class AudioDeepfakeDetector(nn.Module):
    def __init__(self):
        super(AudioDeepfakeDetector, self).__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the final Fully Connected layer for binary classification (Real vs AI)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2) 

    def forward(self, x):
        return self.resnet(x)

def preprocess_audio(audio_path, target_size=(224, 224)):
    audio_path="good morning ai.mp3"
    # 1. Load audio
    y, sr = librosa.load(audio_path, duration=3.0) # Take first 3 seconds
    
    # 2. Generate Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 3. Normalize to 0-255 (Image format)
    img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255
    img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    
    # 4. Resize and Transform for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return preprocess(img).unsqueeze(0) # Add batch dimension

if __name__ == "__main__":
    # 1. Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeDetector().to(device)
    model.eval() # Set to evaluation mode
    
    # 2. Path to the audio file you want to check
    audio_path = r"M:\PROGRAM\audio scanner\sun.wav"
    
    try:
        # 3. Process and Predict
        print(f"Analyzing: {audio_path}...")
        input_tensor = preprocess_audio(audio_path).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item() * 100

        # 4. Output Result
        labels = ["Human/Real", "AI/Fake"]
        print("-" * 30)
        print(f"Result: {labels[prediction]}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error processing file: {e}")

# --- Usage Example ---
# model = AudioDeepfakeDetector()
# model.eval()
# input_tensor = preprocess_audio("path_to_your_audio.wav")
# output = model(input_tensor)
# prediction = torch.argmax(output, dim=1).item()
# print("AI Generated" if prediction == 1 else "Human Voice")
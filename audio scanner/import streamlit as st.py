import torch
import torch.nn as nn
import librosa
import numpy as np
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os

class AudioDeepfakeDetector(nn.Module):
    def __init__(self, pretrained_path=None):
        super(AudioDeepfakeDetector, self).__init__()
        # Use weights=ResNet18_Weights.DEFAULT for modern PyTorch versions
        self.resnet = models.resnet18(weights='DEFAULT')
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2) 
        
        # Load custom trained weights if they exist
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            print(f"Loaded trained weights from {pretrained_path}")
        else:
            print("Warning: Using untrained/random weights. Results will be inaccurate until trained.")

    def forward(self, x):
        return self.resnet(x)

def preprocess_audio(audio_path, target_size=(224, 224)):
    # 1. Load audio (Fixed length: 3 seconds)
    y, sr = librosa.load(audio_path, duration=3.0)
    if len(y) < sr * 3: # Pad if too short
        y = np.pad(y, (0, int(sr * 3) - len(y)))
    
    # 2. Generate Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 3. Normalize to 0-255 (Image format)
    # Added epsilon to avoid division by zero if audio is silent
    img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-6) * 255
    img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
    
    # 4. Transform for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return preprocess(img).unsqueeze(0)

# --- NEW: Simple Training Function ---
def train_one_step(model, audio_path, label, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    input_tensor = preprocess_audio(audio_path).to(device)
    target = torch.tensor([label]).to(device) # 0 for Real, 1 for Fake
    
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_file = "deepfake_detector_weights.pth"
    
    # Initialize model with weight loading
    model = AudioDeepfakeDetector(pretrained_path=weight_file).to(device)
    
    # Example Path
    audio_path = r"M:\PROGRAM\audio scanner\you-loseheavy-echoed-voice-230555.mp3"
    
    # --- MODE SELECTION ---
    # Change this to True if you want to 'teach' the model that 'sun.wav' is Fake (1)
    TRAIN_MODE = False 

    if TRAIN_MODE:
        print("Training mode active...")
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        # Training on one sample as an example (usually you'd loop through a folder)
        loss = train_one_step(model, audio_path, label=1, optimizer=optimizer, criterion=criterion, device=device)
        torch.save(model.state_dict(), weight_file)
        print(f"Training Loss: {loss:.4f}. Model saved.")
    
    else:
        # INFERENCE MODE
        model.eval()
        try:
            input_tensor = preprocess_audio(audio_path).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item() * 100

            labels = ["Human/Real", "AI/Fake"]
            print("-" * 30)
            print(f"Result: {labels[prediction]}")
            print(f"Confidence: {confidence:.2f}%")
            print("-" * 30)
        except Exception as e:
            print(f"Error: {e}")
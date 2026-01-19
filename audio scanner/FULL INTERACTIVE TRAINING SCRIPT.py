import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from torchvision import models, transforms
from PIL import Image

# ------------------------------
# Model Definition
# ------------------------------
class AudioDeepfakeDetector(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        if weight_path and os.path.exists(weight_path):
            self.load_state_dict(torch.load(weight_path, map_location="cpu"))
            print("✅ Loaded existing trained weights")
        else:
            print("⚠️ No trained weights found. Starting fresh.")

    def forward(self, x):
        return self.model(x)


# ------------------------------
# Audio → Mel Spectrogram Image
# ------------------------------
def preprocess_audio(audio_path, duration=3.0):

    y, sr = librosa.load(audio_path, duration=duration)

    if len(y) < sr * duration:
        y = np.pad(y, (0, int(sr * duration) - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    mel_img = (mel_norm * 255).astype(np.uint8)

    img = Image.fromarray(mel_img).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)


# ------------------------------
# Prediction
# ------------------------------
def predict(model, audio_path, device):
    model.eval()
    x = preprocess_audio(audio_path).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    pred = probs.argmax(dim=1).item()
    confidence = probs[0, pred].item() * 100
    return pred, confidence


# ------------------------------
# One-Step Online Training
# ------------------------------
def train_step(model, audio_path, label, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    x = preprocess_audio(audio_path).to(device)
    y = torch.tensor([label]).to(device)

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    return loss.item()


# ------------------------------
# Interactive Learning Loop
# ------------------------------
def interactive_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = "deepfake_detector_weights.pth"

    model = AudioDeepfakeDetector(weight_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    LABELS = ["Human / Real", "AI / Fake"]

    print("\n🎧 AUDIO DEEPFAKE DETECTOR (Interactive Mode)")
    print("Type 'exit' to quit\n")

    while True:
        audio_path = input("📂 Enter audio file path: ").strip()

        if audio_path.lower() == "exit":
            break

        if not os.path.exists(audio_path):
            print("❌ File not found\n")
            continue

        pred, conf = predict(model, audio_path, device)
        print(f"\n🔍 Prediction: {LABELS[pred]}")
        print(f"📊 Confidence: {conf:.2f}%")

        user_input = input(
            "\nIs this prediction correct? "
            "(y = yes / n = no / s = skip training): "
        ).lower()

        if user_input == "y":
            label = pred
        elif user_input == "n":
            label = 1 - pred
        else:
            print("⏭ Skipped training\n")
            continue

        loss = train_step(model, audio_path, label, optimizer, criterion, device)
        torch.save(model.state_dict(), weight_path)

        print(f"🧠 Model updated | Training Loss: {loss:.4f}\n")

    print("\n👋 Session ended. Model saved.")


# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    interactive_loop()

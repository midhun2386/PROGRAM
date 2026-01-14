# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np

# # 'y' is the sound data, 'sr' is the sampling rate 
# audio_path ="midhun.wav"
# y, sr = librosa.load(audio_path)

# #Spectrogram
# plt.figure(figsize=(10, 4))
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')

# #plottings
# plt.colorbar(format='%+2.0f dB')
# plt.title('Voice Fingerprint (Spectrogram)')
# plt.show()

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import soundfile as sf
import logging
from pathlib import Path


audio_path="sun.wav" 
y, sr = librosa.load(audio_path)    


plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')


plt.colorbar(format='%+2.0f dB')
plt.title('Voice Fingerprint (Spectrogram)')
plt.show()

def analyze_audio(audio_path):
    model_id = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)


    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 3. Prepare inputs for the model
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # 4. Perform Inference
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 5. Interpret Results
    # Applying Softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()

    # Label Mapping (Verify specific model labels)
    labels = {0: "Human (Real)", 1: "AI (Fake/Synthetic)"}
    
    print(f"--- Analysis Results for {audio_path} ---")
    print(f"Prediction: {labels[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")

# Run the analysis
analyze_audio(audio_path)
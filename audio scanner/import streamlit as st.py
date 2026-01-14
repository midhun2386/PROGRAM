import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
import soundfile as sf
import os

# 1. SETUP & CACHING (Saves time during judges' demo)
st.set_page_config(page_title="VoiceShield AI", layout="wide")

@st.cache_resource
def load_ai_model():
    # Downloads the model only the first time
    return pipeline("audio-classification", model="mel03/wav2vec2-base-finetuned-deepfake")

# 2. UI HEADER
st.title("🛡️ VoiceShield AI: Deepfake Detection System")
st.markdown("#### Theme: Cyber Security & AI Ethics | Seminar Hall-1")

# 3. SIDEBAR CONTROLS
st.sidebar.header("Project Settings")
st.sidebar.write("This tool uses **Wav2Vec 2.0** to analyze audio frequencies for synthetic patterns.")

# 4. FILE UPLOADER
uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if uploaded_file is not None:
    # Save file locally for processing
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Frequency Analysis")
        # Using the logic that worked in your VS Code
        y, sr = librosa.load("temp.wav")
        fig, ax = plt.subplots(figsize=(10, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        plt.colorbar(img, format='%+2.0f dB')
        st.pyplot(fig)
        st.caption("Human voices show natural variance; AI voices often show rigid patterns.")

    with col2:
        st.subheader("🧠 AI Classification")
        with st.spinner('Scanning for digital artifacts...'):
            detector = load_ai_model()
            results = detector("temp.wav")
            
            label = results[0]['label']
            score = results[0]['score']

            if label.lower() == "bonafide":
                st.success(f"### ✅ RESULT: HUMAN")
                st.write(f"Confidence: {round(score * 100, 2)}%")
            else:
                st.error(f"### 🚨 RESULT: DEEPFAKE")
                st.write(f"Confidence: {round(score * 100, 2)}%")

    st.divider()
    st.write("🏁 **Hackathon Pitch:** This project addresses the rise of 'Vishing' (Voice Phishing) by providing a real-time verification dashboard.")
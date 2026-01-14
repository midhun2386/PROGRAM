from flask import Flask, request, render_template
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "audio" not in request.files:
        return "No file uploaded"

    file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load audio
    y, sr = librosa.load(file_path)

    # Create spectrogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Voice Fingerprint (Spectrogram)")
    plt.tight_layout()
    plt.show()

    return "Audio uploaded and processed successfully"

if __name__ == "__main__":
    app.run(debug=True)

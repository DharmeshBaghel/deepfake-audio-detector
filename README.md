# 🎙️ Deepfake Audio & Voice Forensics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red.svg)](https://streamlit.io/)

## 🔍 Project Overview
This project is a full-stack machine learning web application designed to detect synthetic and AI-generated audio (deepfakes). Using a Long Short-Term Memory (LSTM) neural network, the application analyzes the spectral artifacts and frequency patterns of an uploaded `.wav` file to determine if the voice is an authentic human or an AI clone.

**Live Demo:** [Click here to test the live application!](https://deepfake-audio-detector-n8n6tnxwsbwmt2vdzh3bda.streamlit.app/)

---

## ✨ Core Features
* **🧠 Neural Network Analysis:** Utilizes a custom-trained LSTM model to predict the authenticity of audio based on extracted Mel-frequency cepstral coefficients (MFCCs).
* **📊 Visual Forensics:** Automatically generates and displays high-resolution Waveforms and Mel-Spectrograms so users can physically see the acoustic differences.
* **🗄️ Investigation History:** Features a built-in SQLite database that acts as a secure, persistent ledger of all previously scanned audio files and their verdicts.
* **📄 PDF Reporting:** Includes a custom report generator (`fpdf2`) that allows users to download an official forensic PDF detailing the AI's confidence metrics and visual evidence.

---

## 🛠️ Technology Stack
* **Machine Learning Engine:** TensorFlow / Keras (LSTM Neural Network)
* **Audio Processing:** Librosa (Feature extraction, MFCCs, Spectrograms)
* **Frontend Framework:** Streamlit (Cloud-hosted UI)
* **Data Visualization:** Matplotlib & Pandas
* **Database & Export:** SQLite3, FPDF2

---

## 🚀 How to Run Locally

If you want to run this application on your own machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/deepfake-audio-detector.git](https://github.com/your-username/deepfake-audio-detector.git)
   cd deepfake-audio-detector

2. Install the required dependencies:
   pip install -r requirements.txt

3. Run the Streamlit server:
   python -m streamlit run app.py

4. Open your browser: The app will automatically launch at http://localhost:8501

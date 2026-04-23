# 🕵️‍♂️ Deepfake Audio Detection & Forensics Suite

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Supabase](https://img.shields.io/badge/Database-Supabase-3ECF8E.svg)
![Gemini](https://img.shields.io/badge/AI-Google_Gemini-8E75B2.svg)

## 📌 Project Overview
A full-stack, enterprise-grade forensic application designed to detect synthetic and AI-generated speech. Built with a custom Deep Learning (LSTM) core, this system extracts acoustic anomalies—such as unnatural zero-crossing rates and spectral flatness—and provides human-readable intelligence reports for non-technical investigators.

## 🚀 Enterprise Architecture & Features

### 1. Neural Network Core (Audio Forensics)
* **Librosa Feature Extraction:** Isolates micro-frictions and ambient resonance patterns that synthetic voice engines (like ElevenLabs or VITS) fail to replicate.
* **LSTM Inference Engine:** Evaluates multidimensional audio arrays to generate a high-precision confidence score on human authenticity.
* **Explainable AI (XAI):** Generates saliency heatmaps to visually isolate the exact timestamps where the AI detected synthetic artifacts.

### 2. Full-Stack Data & Security Pipeline
* **Cloud Database Integration:** Synchronizes all forensic scans to a live **Supabase (PostgreSQL)** backend for permanent record keeping.
* **Role-Based Access Control (RBAC):** Secured, password-protected administrative command center.
* **Live System Analytics:** Real-time data visualization of system traffic, confidence trends, and verdict distributions using Pandas and Streamlit charting.
* **Smart Search:** Dynamic dataframe filtering for instant forensic record retrieval.

### 3. Generative AI & MLOps 
* **Plain-English Summaries:** Integrates the **Google Gemini API** to translate complex spectral math into professional, 3-sentence investigation summaries with built-in rate-limit fail-safes.
* **Automated PDF Reporting:** Dynamically compiles the acoustic math, AI summaries, and visual waveform data into court-ready, downloadable PDF reports.
* **MLOps Feedback Loop:** Features a "Human-in-the-Loop" architecture allowing admins to flag False Positives/Negatives directly in the UI, tagging the data for future model retraining.

## 💻 Tech Stack
* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** TensorFlow/Keras, Librosa, NumPy, Matplotlib
* **Database:** Supabase
* **External APIs:** Google Gemini API
* **Document Generation:** FPDF

## ⚙️ Local Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Set up your environment variables locally in a .streamlit/secrets.toml file:
   ```bash
   ADMIN_PASSWORD = "your_secure_password"
   GEMINI_API_KEY = "your_google_api_key"
   SUPABASE_URL = "your_supabase_url"
   SUPABASE_KEY = "your_supabase_key"

4. Run the application:
   ```bash
   streamlit run app.py

## 🛡️ Security & Privacy Note
This tool is designed for educational and defensive cybersecurity purposes. Uploaded audio files are processed entirely in memory and temporarily on disk. To ensure strict data privacy, all raw audio payloads and intermediate visualizations are automatically purged from the local file system immediately following PDF generation.
   

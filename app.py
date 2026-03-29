import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import extract_features
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from datetime import datetime

# --- DATABASE SETUP ---
def init_db():
    """Creates the database table if it doesn't exist yet."""
    conn = sqlite3.connect('forensics_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  confidence REAL,
                  verdict TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_record(filename, confidence, verdict):
    """Saves a new analysis record to the database."""
    conn = sqlite3.connect('forensics_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (filename, confidence, verdict, timestamp) VALUES (?, ?, ?, ?)",
              (filename, float(confidence), verdict, timestamp))
    conn.commit()
    conn.close()

def fetch_history():
    """Retrieves the history as a Pandas DataFrame for easy graphing."""
    conn = sqlite3.connect('forensics_history.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

# Initialize the database when the app starts
init_db()

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="wide")
st.title("🎙️ Deepfake Voice & Audio Detector")

# Create two tabs for the interface
tab1, tab2 = st.tabs(["🔍 Live Scanner", "📊 History Dashboard"])

# Load the trained model
@st.cache_resource
def load_trained_model():
    if os.path.exists("deepfake_audio_model.h5"):
        return load_model("deepfake_audio_model.h5")
    return None

model = load_trained_model()

# ==========================================
# TAB 1: THE LIVE SCANNER
# ==========================================
with tab1:
    st.write("Upload an audio file to determine if it is a real human voice or AI-generated.")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Audio"):
            if model is None:
                st.error("Model not found! Please check your repository.")
            else:
                with st.spinner("Analyzing spectral artifacts and generating visuals..."):
                    temp_file = "temp.wav"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Visualizations
                    st.markdown("### 📊 Audio Forensics Analysis")
                    col1, col2 = st.columns(2)
                    y, sr = librosa.load(temp_file, sr=16000)
                    
                    with col1:
                        st.write("**Waveform (Amplitude over Time)**")
                        fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
                        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#1f77b4")
                        st.pyplot(fig_wave)
                    
                    with col2:
                        st.write("**Mel Spectrogram (Frequencies over Time)**")
                        fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec)
                        st.pyplot(fig_spec)

                    # Prediction
                    features = extract_features(temp_file)
                    if features is not None:
                        features = np.expand_dims(features, axis=0) 
                        prediction = model.predict(features)[0][0]
                        
                        st.markdown("---")
                        st.markdown("### 🤖 Neural Network Verdict")
                        
                        # Determine verdict and save to database
                        if prediction > 0.5:
                            verdict = "FAKE"
                            confidence = prediction * 100
                            st.error(f"🚨 **FAKE AUDIO DETECTED** (Confidence: {confidence:.2f}%)")
                        else:
                            verdict = "REAL"
                            confidence = (1 - prediction) * 100
                            st.success(f"✅ **REAL HUMAN VOICE** (Confidence: {confidence:.2f}%)")
                        
                        # --- THE MAGIC STEP: Save to DB ---
                        save_record(uploaded_file.name, confidence, verdict)
                        
                    else:
                        st.error("Could not process the audio file features.")
                        
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

# ==========================================
# TAB 2: THE HISTORY DASHBOARD
# ==========================================
with tab2:
    st.header("🗄️ Investigation History")
    st.write("A secure log of all previously scanned audio files and their verdicts.")
    
    # Fetch the data
    df = fetch_history()
    
    if not df.empty:
        # Create some top-level metrics
        total_scans = len(df)
        fake_count = len(df[df['verdict'] == 'FAKE'])
        real_count = len(df[df['verdict'] == 'REAL'])
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Files Scanned", total_scans)
        col_m2.metric("Deepfakes Caught", fake_count)
        col_m3.metric("Authentic Voices", real_count)
        
        st.markdown("---")
        
        # Display the interactive table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Display a quick bar chart
        st.markdown("### 📈 Detection Ratio")
        chart_data = df['verdict'].value_counts()
        st.bar_chart(chart_data, color="#ff4b4b")
        
    else:
        st.info("No records found. Head over to the Live Scanner to analyze your first file!")

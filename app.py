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

# ==========================================
# 🔒 SECURITY SYSTEM
# ==========================================
# Change this to your secret password!
APP_PASSWORD = "admin123"

def check_password():
    """Returns `True` if the user has entered the correct password."""
    
    # This nested function runs when the user hits 'Enter' on the password box
    def password_entered():
        if st.session_state["password"] == APP_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Delete password from memory for safety
        else:
            st.session_state["password_correct"] = False

    # Check if they have already proven they know the password in this session
    if "password_correct" not in st.session_state:
        st.markdown("### 🔒 Restricted Access")
        st.write("Please enter the password to access the Forensics Dashboard.")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    
    # Check if they guessed wrong
    elif not st.session_state["password_correct"]:
        st.markdown("### 🔒 Restricted Access")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("🚨 Incorrect password. Access denied.")
        return False
    
    # If they passed both checks, let them in!
    return True

# Set up the web page
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="wide")

# --- THE GATEKEEPER ---
if not check_password():
    st.stop()  # Do not run a single line of code below this if the password is wrong!

# ==========================================
# 🗄️ DATABASE SETUP
# ==========================================
def init_db():
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
    conn = sqlite3.connect('forensics_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history (filename, confidence, verdict, timestamp) VALUES (?, ?, ?, ?)",
              (filename, float(confidence), verdict, timestamp))
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect('forensics_history.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_db()

# ==========================================
# 🖥️ MAIN APP LOGIC (Only runs if unlocked)
# ==========================================
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🎙️ Deepfake Voice & Audio Detector")
with col2:
    # Add a quick Log Out button to the top right
    if st.button("Logout 🚪"):
        st.session_state["password_correct"] = False
        st.rerun()

tab1, tab2 = st.tabs(["🔍 Live Scanner", "📊 History Dashboard"])

@st.cache_resource
def load_trained_model():
    if os.path.exists("deepfake_audio_model.h5"):
        return load_model("deepfake_audio_model.h5")
    return None

model = load_trained_model()

# --- TAB 1: LIVE SCANNER ---
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
                    
                    st.markdown("### 📊 Audio Forensics Analysis")
                    c1, c2 = st.columns(2)
                    y, sr = librosa.load(temp_file, sr=16000)
                    
                    with c1:
                        st.write("**Waveform**")
                        fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
                        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#1f77b4")
                        st.pyplot(fig_wave)
                    
                    with c2:
                        st.write("**Mel Spectrogram**")
                        fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec)
                        st.pyplot(fig_spec)

                    features = extract_features(temp_file)
                    if features is not None:
                        features = np.expand_dims(features, axis=0) 
                        prediction = model.predict(features)[0][0]
                        
                        st.markdown("---")
                        st.markdown("### 🤖 Neural Network Verdict")
                        
                        if prediction > 0.5:
                            verdict = "FAKE"
                            confidence = prediction * 100
                            st.error(f"🚨 **FAKE AUDIO DETECTED** (Confidence: {confidence:.2f}%)")
                        else:
                            verdict = "REAL"
                            confidence = (1 - prediction) * 100
                            st.success(f"✅ **REAL HUMAN VOICE** (Confidence: {confidence:.2f}%)")
                        
                        save_record(uploaded_file.name, confidence, verdict)
                    else:
                        st.error("Could not process the audio file features.")
                        
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

# --- TAB 2: HISTORY DASHBOARD ---
with tab2:
    st.header("🗄️ Investigation History")
    df = fetch_history()
    
    if not df.empty:
        total_scans = len(df)
        fake_count = len(df[df['verdict'] == 'FAKE'])
        real_count = len(df[df['verdict'] == 'REAL'])
        
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Total Scans", total_scans)
        cm2.metric("Deepfakes Caught", fake_count)
        cm3.metric("Authentic Voices", real_count)
        
        st.markdown("---")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### 📈 Detection Ratio")
        st.bar_chart(df['verdict'].value_counts(), color="#ff4b4b")
    else:
        st.info("No records found. Run a scan in the Live Scanner to see history!")

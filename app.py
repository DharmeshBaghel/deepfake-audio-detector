import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import extract_features
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from tinytag import TinyTag

# ==========================================
# 🧠 EXPLAINABLE AI (Saliency Map for LSTMs)
# ==========================================
def make_saliency_heatmap(input_features, model):
    """Calculates how much each timeframe impacted the final prediction."""
    input_tensor = tf.convert_to_tensor(input_features, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        preds = model(input_tensor)
        score = preds[0][0]

    grads = tape.gradient(score, input_tensor)
    saliency = tf.reduce_mean(tf.abs(grads), axis=-1)
    saliency = tf.squeeze(saliency)
    
    max_val = tf.math.reduce_max(saliency)
    if max_val > 0:
        saliency = saliency / max_val
        
    return saliency.numpy()

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
# 📄 PDF GENERATOR
# ==========================================
def create_pdf_report(filename, verdict, confidence, wave_path, spec_path, cam_path=None):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", style="B", size=20)
    pdf.cell(200, 15, txt="Audio Forensics Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Target File: {filename}", ln=True)
    pdf.cell(200, 10, txt=f"Date of Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", style="B", size=16)
    if verdict == "FAKE":
        pdf.set_text_color(220, 53, 69)
        pdf.cell(200, 10, txt=f"VERDICT: SYNTHETIC AUDIO DETECTED", ln=True)
    else:
        pdf.set_text_color(40, 167, 69)
        pdf.cell(200, 10, txt=f"VERDICT: AUTHENTIC HUMAN VOICE", ln=True)
        
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Neural Network Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(200, 10, txt="Visual Spectral Evidence:", ln=True)
    
    if os.path.exists(wave_path):
        pdf.image(wave_path, x=10, y=None, w=190)
    if os.path.exists(spec_path):
        pdf.image(spec_path, x=10, y=None, w=190)
    
    if cam_path and os.path.exists(cam_path):
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(200, 10, txt="Explainable AI (Saliency Map):", ln=True)
        pdf.image(cam_path, x=10, y=None, w=190)
        
    report_path = "forensic_report.pdf"
    pdf.output(report_path)
    return report_path

# ==========================================
# 🖥️ MAIN APP LOGIC
# ==========================================
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="wide")
st.title("🎙️ Deepfake Voice & Audio Detector")

tab1, tab2 = st.tabs(["🔍 Live Scanner", "📊 History Dashboard"])

@st.cache_resource
def load_trained_model():
    if os.path.exists("deepfake_audio_model.h5"):
        return load_model("deepfake_audio_model.h5")
    return None

model = load_trained_model()

with tab1:
    st.write("Upload an audio file to determine if it is a real human voice or AI-generated.")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Audio"):
            if model is None:
                st.error("Model not found! Please check your repository.")
            else:
                with st.spinner("Analyzing spectral artifacts and generating XAI heatmaps..."):
                    temp_file = "temp.wav"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # ==========================================
                    # 🕵️ ADVANCED FORENSICS SUITE
                    # ==========================================
                    st.markdown("### 🕵️ Digital Forensics Report")
                    
                    # 1. Hidden Metadata Extraction
                    try:
                        tag = TinyTag.get(temp_file)
                        st.write("**File Metadata (Digital Fingerprint):**")
                        
                        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
                        meta_c1.metric("Sample Rate", f"{tag.samplerate} Hz")
                        meta_c2.metric("Bitrate", f"{tag.bitrate} kbps" if tag.bitrate else "Unknown")
                        meta_c3.metric("Duration", f"{tag.duration:.2f} sec")
                        meta_c4.metric("File Size", f"{tag.filesize / 1024:.1f} KB")
                        
                        if tag.extra:
                            st.caption(f"Hidden Tags Detected: {tag.extra}")
                    except Exception as e:
                        st.warning("Could not extract hidden metadata from this file.")

                    # 2. Acoustic Anomaly Breakdown (Sub-metrics)
                    st.write("**Acoustic Feature Analysis:**")
                    y, sr = librosa.load(temp_file, sr=16000)
                    
                    zcr = librosa.feature.zero_crossing_rate(y)
                    zcr_mean = np.mean(zcr)
                    
                    flatness = librosa.feature.spectral_flatness(y=y)
                    flat_mean = np.mean(flatness)
                    
                    feat_c1, feat_c2 = st.columns(2)
                    
                    with feat_c1:
                        st.write("⏱️ **Vocal Roughness (Zero-Crossing)**")
                        st.progress(min(int(zcr_mean * 1000), 100))
                        if zcr_mean < 0.05:
                            st.caption("🚨 Suspiciously smooth. Human vocal cords usually produce more friction/noise.")
                        else:
                            st.caption("✅ Natural friction detected.")
                            
                    with feat_c2:
                        st.write("🌬️ **Breathiness (Spectral Flatness)**")
                        st.progress(min(int(flat_mean * 10000), 100))
                        if flat_mean < 0.001:
                            st.caption("🚨 Suspiciously pure tone. Lacks natural human breath resonance.")
                        else:
                            st.caption("✅ Natural ambient resonance detected.")
                            
                    st.markdown("---")

                    # ==========================================
                    # 📊 VISUAL SPECTROGRAMS
                    # ==========================================
                    st.markdown("### 📊 Audio Forensics Analysis")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.write("**Waveform**")
                        fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
                        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#1f77b4")
                        st.pyplot(fig_wave)
                        fig_wave.savefig("temp_wave.png", bbox_inches='tight')
                    
                    with c2:
                        st.write("**Mel Spectrogram**")
                        fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        S_dB = librosa.power_to_db(S, ref=np.max)
                        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec)
                        st.pyplot(fig_spec)
                        fig_spec.savefig("temp_spec.png", bbox_inches='tight')

                    features = extract_features(temp_file)
                    if features is not None:
                        features = np.expand_dims(features, axis=0) 
                        prediction = model.predict(features)[0][0]
                        
                        st.markdown("---")
                        
                        # ==========================================
                        # 🔦 EXPLAINABLE AI SECTION
                        # ==========================================
                        st.markdown("### 🔦 Explainable AI (X-Ray Vision)")
                        cam_path = None
                        
                        try:
                            heatmap = make_saliency_heatmap(features, model)
                            
                            time_axis = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(heatmap))
                            fig_cam, ax_cam = plt.subplots(figsize=(10, 2.5))
                            ax_cam.plot(time_axis, heatmap, color='r', linewidth=2)
                            ax_cam.fill_between(time_axis, heatmap, color='r', alpha=0.3)
                            ax_cam.set_xlim([0, time_axis[-1]])
                            ax_cam.set_xlabel("Time (seconds)")
                            ax_cam.set_ylabel("AI Attention Score")
                            
                            st.write("This graph tracks the LSTM's internal focus using **Saliency Mapping**. **Red spikes** indicate the exact timestamps where the AI detected the highest density of synthetic acoustic anomalies.")
                            st.pyplot(fig_cam)
                            fig_cam.savefig("temp_cam.png", bbox_inches='tight')
                            cam_path = "temp_cam.png"
                            
                        except Exception as e:
                            st.warning(f"Could not generate XAI Heatmap: {e}")
                        
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
                        
                        report_file = create_pdf_report(uploaded_file.name, verdict, confidence, "temp_wave.png", "temp_spec.png", cam_path)
                        
                        with open(report_file, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                            
                        st.download_button(
                            label="📄 Download Official Forensic Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"Report_{uploaded_file.name}.pdf",
                            mime="application/pdf"
                        )
                        
                    else:
                        st.error("Could not process the audio file features.")
                        
                    for file in [temp_file, "temp_wave.png", "temp_spec.png", "temp_cam.png", "forensic_report.pdf"]:
                        if os.path.exists(file):
                            os.remove(file)

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
        st.dataframe(df, width='stretch', hide_index=True)
        st.markdown("### 📈 Detection Ratio")
        st.bar_chart(df['verdict'].value_counts(), color="#ff4b4b")
    else:
        st.info("No records found. Run a scan in the Live Scanner to see history!")

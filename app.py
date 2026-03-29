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

# ==========================================
# 🧠 EXPLAINABLE AI (Grad-CAM)
# ==========================================
def get_last_conv_layer_name(model):
    """Dynamically finds the last convolutional layer in the architecture."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Calculates the gradient of the top predicted class to find what the AI focused on."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Pool gradients over the temporal dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Multiply feature map by importance and sum to get the heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

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
        pdf.cell(200, 10, txt="Explainable AI (Grad-CAM Attention Map):", ln=True)
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
                    
                    st.markdown("### 📊 Audio Forensics Analysis")
                    c1, c2 = st.columns(2)
                    y, sr = librosa.load(temp_file, sr=16000)
                    
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
                        
                        # --- EXPLAINABLE AI SECTION ---
                        st.markdown("### 🔦 Explainable AI (X-Ray Vision)")
                        cam_path = None
                        last_conv_layer = get_last_conv_layer_name(model)
                        
                        if last_conv_layer:
                            try:
                                heatmap = make_gradcam_heatmap(features, model, last_conv_layer)
                                
                                # Plot the AI's attention over time
                                time_axis = np.linspace(0, librosa.get_duration(y=y, sr=sr), len(heatmap))
                                fig_cam, ax_cam = plt.subplots(figsize=(10, 2.5))
                                ax_cam.plot(time_axis, heatmap, color='r', linewidth=2)
                                ax_cam.fill_between(time_axis, heatmap, color='r', alpha=0.3)
                                ax_cam.set_xlim([0, time_axis[-1]])
                                ax_cam.set_xlabel("Time (seconds)")
                                ax_cam.set_ylabel("AI Attention Score")
                                
                                st.write("This graph tracks the neural network's internal focus. **Red spikes** indicate the exact timestamps where the AI detected the highest density of synthetic acoustic anomalies.")
                                st.pyplot(fig_cam)
                                fig_cam.savefig("temp_cam.png", bbox_inches='tight')
                                cam_path = "temp_cam.png"
                                
                            except Exception as e:
                                st.warning(f"Could not generate XAI Heatmap: {e}")
                        else:
                            st.info("Grad-CAM requires a Convolutional layer. No Conv layer detected in model architecture.")
                        
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
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("### 📈 Detection Ratio")
        st.bar_chart(df['verdict'].value_counts(), color="#ff4b4b")
    else:
        st.info("No records found. Run a scan in the Live Scanner to see history!")

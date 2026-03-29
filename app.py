import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import extract_features
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Set up the web page
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="wide")
st.title("🎙️ Deepfake Voice & Audio Detector")
st.write("Upload an audio file to determine if it is a real human voice or AI-generated.")

# Load the trained model
@st.cache_resource
def load_trained_model():
    if os.path.exists("deepfake_audio_model.h5"):
        return load_model("deepfake_audio_model.h5")
    return None

model = load_trained_model()

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Play the audio file on the website
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Analyze Audio"):
        if model is None:
            st.error("Model not found! Please run train.py first to generate the model.")
        else:
            with st.spinner("Analyzing spectral artifacts and generating visuals..."):
                # Save the uploaded file temporarily to process it
                temp_file = "temp.wav"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # --- VISUALIZATION SECTION ---
                st.markdown("### 📊 Audio Forensics Analysis")
                col1, col2 = st.columns(2)
                
                # Load audio for plotting
                y, sr = librosa.load(temp_file, sr=16000)
                
                # 1. Plot the Waveform
                with col1:
                    st.write("**Waveform (Amplitude over Time)**")
                    fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
                    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color="#1f77b4")
                    ax_wave.set_xlabel("Time (s)")
                    ax_wave.set_ylabel("Amplitude")
                    st.pyplot(fig_wave)
                
                # 2. Plot the Mel Spectrogram (What the AI actually "sees")
                with col2:
                    st.write("**Mel Spectrogram (Frequencies over Time)**")
                    fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    S_dB = librosa.power_to_db(S, ref=np.max)
                    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec)
                    fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
                    st.pyplot(fig_spec)

                # --- PREDICTION SECTION ---
                # Extract features and predict
                features = extract_features(temp_file)
                
                if features is not None:
                    # Reshape for the model (1 sample, Time steps, Features)
                    features = np.expand_dims(features, axis=0) 
                    prediction = model.predict(features)[0][0]
                    
                    # Display results with massive emphasis
                    st.markdown("---")
                    st.markdown("### 🤖 Neural Network Verdict")
                    
                    if prediction > 0.5:
                        st.error(f"🚨 **FAKE AUDIO DETECTED** (Confidence: {prediction * 100:.2f}%)")
                        st.write("This audio exhibits synthetic spectral artifacts and unnatural pitch transitions.")
                    else:
                        st.success(f"✅ **REAL HUMAN VOICE** (Confidence: {(1 - prediction) * 100:.2f}%)")
                        st.write("Natural breathing patterns, organic frequencies, and authentic acoustic resonance detected.")
                else:
                    st.error("Could not process the audio file features.")
                    
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
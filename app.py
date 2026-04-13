import google.generativeai as genai
from supabase import create_client, Client
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import extract_features
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
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
# 🗄️ CLOUD DATABASE SETUP (Supabase API)
# ==========================================
# Securely fetch the API keys from Streamlit's vault
url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

def save_record(filename, confidence, verdict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "filename": filename,
        "confidence": float(confidence),
        "verdict": verdict,
        "timestamp": timestamp
    }
    # Insert data via secure API instead of SQL
    supabase.table('history').insert(data).execute()

def fetch_history():
    # Fetch data via secure API
    response = supabase.table('history').select('*').order('timestamp', desc=True).execute()
    # Convert directly to a pandas dataframe
    return pd.DataFrame(response.data)

# Note: We no longer need init_db() because we built the table in the Supabase UI

#==========================================
# 🤖 GEMINI AI SETUP
# ==========================================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

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

# ==========================================
# 🧠 APP MEMORY (Session State)
# ==========================================
# Initialize the admin login state so it exists globally
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
    
st.title("🎙️ Deepfake Voice & Audio Detector")

tab1, tab2 = st.tabs(["🔍 Scan Audio", "⚙️ Admin & History"])

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
                    # 🧠 1. CORE MATH & AI PREDICTION (Invisible Backend)
                    # ==========================================
                    # Load audio
                    y, sr = librosa.load(temp_file, sr=16000)
                    
                    # Calculate math
                    zcr = librosa.feature.zero_crossing_rate(y)
                    zcr_mean = np.mean(zcr)
                    
                    flatness = librosa.feature.spectral_flatness(y=y)
                    flat_mean = np.mean(flatness)
                    
                    # Run LSTM AI Prediction EARLY
                    features = extract_features(temp_file)
                    if features is not None:
                        features = np.expand_dims(features, axis=0) 
                        prediction = model.predict(features)[0][0]
                        
                        # Define the verdict before drawing the UI
                        if prediction > 0.5:
                            verdict = "FAKE"
                            confidence = prediction * 100
                        else:
                            verdict = "REAL"
                            confidence = (1 - prediction) * 100
                        
                        # ==========================================
                        # 🖥️ 2. UI DASHBOARD RENDERING
                        # ==========================================
                        st.markdown("### 🕵️ Digital Forensics Report")
                        
                        # --- 1. NEURAL NETWORK VERDICT ---
                        if verdict == "FAKE":
                            st.error(f"🚨 **FAKE AUDIO DETECTED** (Confidence: {confidence:.2f}%)")
                        else:
                            st.success(f"✅ **REAL HUMAN VOICE** (Confidence: {confidence:.2f}%)")
                        st.markdown("---")
                        
                        # --- 2. HIDDEN METADATA ---
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

                        st.markdown("---")

                        # --- 3. DYNAMIC ACOUSTIC FEATURES (Red/Green Logic) ---
                        st.write("**Acoustic Feature Analysis:**")
                        
                        # 1. Define dynamic messages AND one-line explanations
                        if verdict == "FAKE":
                            zcr_message = "⚠️ Unnatural glitches"
                            flat_message = "⚠️ Artificial resonance"
                            text_color = "inverse"
                            
                            # The new one-line explanations for FAKE audio
                            zcr_explanation = "AI models struggle to copy human vocal cord friction, creating a suspiciously smooth signal."
                            flat_explanation = "This audio lacks natural breathing frequencies, sounding too 'pure' to be human."
                        else:
                            zcr_message = "✅ Natural friction"
                            flat_message = "✅ Natural ambient tone"
                            text_color = "normal"
                            
                            # The new one-line explanations for REAL audio
                            zcr_explanation = "Contains the natural micro-frictions and variations expected from real human vocal cords."
                            flat_explanation = "Natural ambient resonance and human breath patterns are clearly present."
                            
                        # 2. Draw the UI
                        feat_c1, feat_c2 = st.columns(2)
                        
                        with feat_c1:
                            st.metric(
                                label="Vocal Roughness (Zero-Crossing)", 
                                value=f"{zcr_mean:.4f}", 
                                delta=zcr_message,
                                delta_color=text_color
                            )
                            # Add the one-line explanation right under the number
                            st.caption(f"*{zcr_explanation}*")
                            
                        with feat_c2:
                            st.metric(
                                label="Breathiness (Spectral Flatness)", 
                                value=f"{flat_mean:.6f}", 
                                delta=flat_message,
                                delta_color=text_color
                            )
                            # Add the one-line explanation right under the number
                            st.caption(f"*{flat_explanation}*")
                            
                        st.markdown("---")

                        # --- 4. VISUAL SPECTROGRAMS ---
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

                        st.markdown("---")
                        
                        # --- 5. EXPLAINABLE AI SECTION ---
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
                        
                        # --- 6. AI INVESTIGATOR SUMMARY (GEMINI) ---
                        st.markdown("### 🕵️ AI Investigator Summary")
                        with st.spinner("Writing simple summary..."):
                            try:
                                prompt = f"""
                                You are a digital forensics expert. 
                                I just scanned an audio file. Here are the results:
                                - Verdict: {verdict}
                                - AI Score: {confidence:.2f}% FAKE
                                - Voice Roughness (Zero-Crossing): {zcr_mean:.4f}
                                - Breathiness (Flatness): {flat_mean:.6f}
                                
                                Write a simple 3-sentence summary for a non-technical person. 
                                Explain what these numbers mean and why the audio is {verdict}. 
                                Keep it professional.
                                """
                                summary_response = gemini_model.generate_content(prompt)
                                st.info(summary_response.text)
                            except Exception as e:
                                st.warning(f"Could not load AI summary at this time. Error: {e}")
                        
                        # --- 7. DATABASE SAVE & REPORT ---
                        save_record(uploaded_file.name, confidence, verdict)
                        
                        try:
                            report_file = create_pdf_report(uploaded_file.name, verdict, confidence, "temp_wave.png", "temp_spec.png", cam_path)
                            with open(report_file, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                                
                            st.download_button(
                                label="📄 Download Official Forensic Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"Report_{uploaded_file.name}.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"Could not generate PDF: {e}")
                            
                    else:
                        st.error("Could not process the audio file features.")
                        
                    # --- 8. SYSTEM CLEANUP ---
                    for file in [temp_file, "temp_wave.png", "temp_spec.png", "temp_cam.png", "forensic_report.pdf"]:
                        if os.path.exists(file):
                            os.remove(file)
                            
with tab2:
    st.header("⚙️ Admin & History Center")
    
    # 1. Show Login Screen if Locked
    if not st.session_state.admin_logged_in:
        st.markdown("🔒 Restricted access. Please log in to view analytics and scan history.")
        admin_pass = st.text_input("Enter Admin Password", type="password")
        
        enter_btn = st.button("Enter 🔐", type="primary")
        
        if enter_btn:
            if admin_pass == st.secrets["ADMIN_PASSWORD"]:
                st.session_state.admin_logged_in = True
                st.rerun() 
            else:
                st.error("Access Denied. Incorrect password.")

    # 2. Show Unified Dashboard if Unlocked
    if st.session_state.admin_logged_in:
        
        if st.button("Logout 🚪"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.success("Authentication successful. Welcome, Admin.")
        st.markdown("---")
        
        # Grab the live cloud data
        df = fetch_history()
        
        if not df.empty:
            # ==========================================
            # 📊 LIVE SYSTEM ANALYTICS
            # ==========================================
            st.subheader("Live System Analytics")
            
            # 1. Calculate the math
            total_scans = len(df)
            fake_count = len(df[df['verdict'] == 'FAKE'])
            real_count = len(df[df['verdict'] == 'REAL'])
            avg_confidence = df['confidence'].mean()
            
            # Find scans from today
            today_date = datetime.now().strftime("%Y-%m-%d")
            scans_today = len(df[df['timestamp'].str.contains(today_date)])
            
            # 2. Top Row Metrics (The Big Numbers)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Scans", total_scans)
            col2.metric("Fakes Caught", fake_count)
            col3.metric("Real Audio", real_count)
            
            # 3. Bottom Row Metrics (Extra Details)
            col4, col5, col6 = st.columns(3)
            col4.metric("Avg AI Certainty", f"{avg_confidence:.1f}%")
            col5.metric("Scans Today", scans_today)
            col6.metric("Database Status", "Online 🟢")
            
            st.markdown("---")
            
            # ==========================================
            # 📈 VISUAL DATA CHARTS
            # ==========================================
            st.subheader("Data Charts")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Real vs Fake Total**")
                chart_data = df['verdict'].value_counts()
                st.bar_chart(chart_data, color="#ff4b4b") 
                
            with chart_col2:
                st.markdown("**AI Certainty Trend**")
                # Create a line chart using timestamp as the X-axis and confidence as the Y-axis
                trend_data = df.set_index('timestamp')['confidence']
                st.line_chart(trend_data, color="#0068c9")
                
            st.markdown("---")
            
            # ==========================================
            # 🗂️ FULL SCAN HISTORY & SMART SEARCH
            # ==========================================
            st.subheader("🗂️ Full Scan History")
            
            # 1. Build the Search UI
            search_col1, search_col2 = st.columns([2, 1]) # Make the text box wider than the dropdown
            
            with search_col1:
                # Text input for searching filenames
                text_query = st.text_input("🔍 Search by Filename", placeholder="e.g., suspect_audio.wav")
                
            with search_col2:
                # Dropdown menu to filter by verdict
                verdict_filter = st.selectbox("🚦 Filter by Verdict", ["All Scans", "FAKE", "REAL"])
            
            # 2. Filter the Data (The "Smart" part)
            filtered_df = df.copy() # Make a safe copy of your cloud data
            
            # If they typed something, filter the filename column
            if text_query:
                # case=False means 'A' and 'a' are treated the same
                filtered_df = filtered_df[filtered_df['filename'].str.contains(text_query, case=False, na=False)]
                
            # If they used the dropdown, filter the verdict column
            if verdict_filter != "All Scans":
                filtered_df = filtered_df[filtered_df['verdict'] == verdict_filter]
            
            # 3. Display the results
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
                
                # 4. Smart CSV Export (Only downloads what is on the screen!)
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download {len(filtered_df)} Filtered Logs (CSV)",
                    data=csv_data,
                    file_name='deepfake_filtered_logs.csv',
                    mime='text/csv',
                    type="primary"
                )
            else:
                st.warning("No scans match your search criteria. Try a different filename!")

            # ==========================================
            # 🛠️ MLOPS: AI FEEDBACK LOOP
            # ==========================================
            st.markdown("---")
            st.subheader("🛠️ AI Model Correction")
            st.markdown("Did the AI make a mistake? Flag false positives or false negatives below to save the audio for future model retraining.")
            
            # Create an expander to keep the UI clean
            with st.expander("Flag an Incorrect Scan"):
                # Get the IDs currently visible on the screen
                available_ids = filtered_df['id'].tolist()
                
                if available_ids:
                    # Layout the form
                    form_col1, form_col2 = st.columns(2)
                    
                    with form_col1:
                        # Dropdown to pick the specific scan
                        selected_id = st.selectbox("Select Scan ID to Correct:", available_ids)
                        
                    with form_col2:
                        # Dropdown to select what the audio ACTUALLY is
                        true_label = st.selectbox("What is the TRUE label of this audio?", ["Actually REAL", "Actually FAKE"])
                        
                    # Submit button
                    if st.button("Flag for Retraining 🚩", type="primary"):
                        try:
                            # 1. Send the correction to Supabase
                            supabase.table("history").update({"human_correction": true_label}).eq("id", selected_id).execute()
                            
                            # 2. Show success message
                            st.success(f"Successfully flagged Scan ID {selected_id}! The data engineering team can now use this file to retrain the model.")
                            
                            # 3. Refresh the app to show the updated database
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Failed to update database. Error: {e}")
                else:
                    st.info("No scans available to correct.")
                
        else:
            st.info("The database is currently empty. Run a scan to see history and analytics!")



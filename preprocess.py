import librosa
import numpy as np

def extract_features(file_path, max_pad_len=400):
    """
    Loads an audio file and extracts its MFCC features.
    """
    try:
        # Load the audio file (resampling to 16kHz is standard for speech)
        audio, sample_rate = librosa.load(file_path, sr=16000)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Pad or truncate the audio to ensure uniform length for the neural network
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
            
        return mfccs.T # Transpose for the LSTM model (Time Steps, Features)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
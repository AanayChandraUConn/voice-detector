import streamlit as st
import librosa
import numpy as np
from tensorflow import keras
import tempfile

TARGET_SHAPE = (128, 63)

st.set_page_config(page_title="AI Voice Detector", page_icon="🎙️", layout="centered")

st.title("🎙️ AI Voice Detector")
st.caption("Built by Aanay Chandra · [GitHub repo](https://github.com/AanayChandraUConn/voice-detector)")

st.write("Upload a short audio clip and this'll guess whether it's a real human voice or an AI-generated one.")

with st.expander("How this works"):
    st.write("""
    This uses a CNN (convolutional neural network) trained on ~700 audio clips,
    half real human voices and half AI-generated ones. Each clip gets turned into
    a mel spectrogram (basically a picture of the sound over time), and the model
    looks for patterns that tend to show up differently in AI-generated speech.
    """)

@st.cache_resource
def load_model():
    return keras.models.load_model("models/cnn_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    col1, col2 = st.columns([2, 1])
    with col1:
        st.audio(uploaded_file)

    with st.spinner("Analyzing the audio..."):
        audio, sr = librosa.load(tmp_path, sr=16000)
        spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)

        if spec_db.shape[1] < TARGET_SHAPE[1]:
            pad_width = TARGET_SHAPE[1] - spec_db.shape[1]
            spec_db = np.pad(spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec_db = spec_db[:, :TARGET_SHAPE[1]]

        spec_db = spec_db[np.newaxis, ..., np.newaxis]

        prediction = model.predict(spec_db)
        confidence = float(prediction[0][0])

    st.divider()
    st.subheader("Result")

    if confidence > 0.5:
        st.success(f"### ✅ Sounds REAL\n**{confidence:.1%}** confidence")
        st.progress(confidence)
    else:
        fake_confidence = 1 - confidence
        st.error(f"### 🤖 Sounds AI-GENERATED\n**{fake_confidence:.1%}** confidence")
        st.progress(fake_confidence)

    st.caption("Note: trained on one dataset, so it may not generalize perfectly to every AI voice generator out there.")

else:
    st.info("👆 Upload a clip to get started")

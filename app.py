import streamlit as st
import librosa
import numpy as np
from tensorflow import keras
import tempfile

TARGET_SHAPE = (128, 63)

st.set_page_config(page_title="AI Voice Detector", layout="centered")

# using a nicer font than the streamlit default
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    margin-top: 10px;
}
.result-real {
    background-color: #ecfdf5;
    border-left: 6px solid #10b981;
}
.result-fake {
    background-color: #fef2f2;
    border-left: 6px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Voice Detector")

@st.cache_resource
def load_model():
    return keras.models.load_model("models/cnn_model.keras")

model = load_model()

tab1, tab2 = st.tabs(["Try it", "How it works"])

with tab1:
    st.write("Upload a short audio clip and this'll guess whether it's a real human voice or an AI-generated one.")

    uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

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

        if confidence > 0.5:
            st.markdown(f"""
            <div class="result-box result-real">
                <h3>Sounds REAL</h3>
                <p>{confidence:.1%} confidence</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence)
        else:
            fake_confidence = 1 - confidence
            st.markdown(f"""
            <div class="result-box result-fake">
                <h3>Sounds AI-GENERATED</h3>
                <p>{fake_confidence:.1%} confidence</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(fake_confidence)

        st.caption("Note: trained on one dataset, so it may not generalize perfectly to every AI voice generator out there.")

    else:
        st.info("Upload a clip to get started")

with tab2:
    st.write("""
    This uses a CNN (convolutional neural network) trained on ~700 audio clips,
    half real human voices and half AI-generated ones. Each clip gets turned into
    a mel spectrogram (basically a picture of the sound over time), and the model
    looks for patterns that tend to show up differently in AI-generated speech.
    """)

    st.write("**Some numbers:**")
    st.write("- 92.14% accuracy on the test set")
    st.write("- Trained on 700 clips (350 real, 350 AI-generated)")
    st.write("- Dataset: Fake-or-Real (FoR) from Kaggle")

    st.write("Built by Aanay Chandra · [GitHub repo](https://github.com/AanayChandraUConn/voice-detector)")

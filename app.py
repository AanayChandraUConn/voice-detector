import streamlit as st
import librosa
import numpy as np
from tensorflow import keras
import tempfile

TARGET_SHAPE = (128, 63)

st.set_page_config(page_title="AI Voice Detector", layout="centered")

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(ellipse 900px 420px at 50% -10%, rgba(139, 92, 246, 0.16), transparent 70%),
            linear-gradient(180deg, #0d0d12 0%, #131318 100%);
    }
    .block-container {
        max-width: 720px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }
    .hero {
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .hero .hero-caption {
        opacity: 0.6;
        font-size: 0.95rem;
        margin-bottom: 1.1rem;
    }
    .hero .hero-caption a {
        color: #a78bfa;
    }
    .hero .hero-sub {
        font-size: 1.05rem;
        opacity: 0.85;
        max-width: 480px;
        margin: 0 auto;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1a22;
        border: 1.5px dashed #3a3a46;
        border-radius: 16px;
    }
    [data-testid="stExpander"] {
        background-color: #16161c;
        border: 1px solid #26262e;
        border-radius: 14px;
    }
    [data-testid="stAlertContainer"] {
        background-color: #16161c;
        border: 1px solid #26262e;
        border-left: 3px solid #8b5cf6;
        border-radius: 12px;
    }
    hr {
        border-color: #26262e;
    }
    .result-card {
        border-radius: 18px;
        padding: 1.75rem 2rem;
        margin-top: 1rem;
        background: #16161c;
    }
    .result-card.real { border-left: 4px solid #34d399; }
    .result-card.fake { border-left: 4px solid #f87171; }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.6;
        margin-bottom: 0.25rem;
    }
    .result-verdict {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .result-verdict.real { color: #34d399; }
    .result-verdict.fake { color: #f87171; }
    .confidence-track {
        height: 8px;
        border-radius: 999px;
        background: #26262e;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
    }
    .confidence-fill.real { background: linear-gradient(90deg, #10b981, #34d399); }
    .confidence-fill.fake { background: linear-gradient(90deg, #ef4444, #f87171); }
    .confidence-pct {
        margin-top: 0.5rem;
        font-size: 0.95rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>AI Voice Detector</h1>
    <p class="hero-caption">Built by Aanay Chandra · <a href="https://github.com/AanayChandraUConn/voice-detector" target="_blank">GitHub repo</a></p>
    <p class="hero-sub">Upload a short audio clip and this'll guess whether it's a real human voice or an AI-generated one.</p>
</div>
""", unsafe_allow_html=True)

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
        verdict, pct, css_class = "Sounds real", confidence, "real"
    else:
        verdict, pct, css_class = "Sounds AI-generated", 1 - confidence, "fake"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">Result</div>
        <div class="result-verdict {css_class}">{verdict}</div>
        <div class="confidence-track">
            <div class="confidence-fill {css_class}" style="width: {pct * 100:.1f}%;"></div>
        </div>
        <div class="confidence-pct">{pct:.1%} confidence</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Note: trained on one dataset, so it may not generalize perfectly to every AI voice generator out there.")

else:
    st.info("Upload a clip to get started")

import streamlit as st
import librosa
import numpy as np
from tensorflow import keras
import tempfile

TARGET_SHAPE = (128, 63)

st.set_page_config(page_title="AI Voice Detector", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbarActions"], #MainMenu {
        display: none;
    }

    .stApp {
        background:
            radial-gradient(ellipse 900px 500px at 10% -10%, rgba(139, 92, 246, 0.18), transparent 65%),
            radial-gradient(ellipse 700px 400px at 100% 5%, rgba(56, 189, 248, 0.10), transparent 60%),
            linear-gradient(180deg, #0b0b10 0%, #121218 100%);
    }
    .block-container {
        max-width: 800px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, .nav-word, .section-label {
        font-family: 'Space Grotesk', sans-serif;
    }

    .site-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 2rem;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .nav-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 9px;
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
        color: #fff;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .nav-word {
        font-weight: 600;
        font-size: 1.05rem;
        color: #f0f0f5;
    }
    a.nav-link, a.nav-link:visited {
        display: inline-block;
        color: #d4d4dd !important;
        text-decoration: none !important;
        font-size: 0.88rem;
        border: 1px solid rgba(255, 255, 255, 0.16);
        padding: 0.4rem 0.95rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.04);
    }

    .hero h1 {
        font-size: 2.5rem;
        line-height: 1.18;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f5f5f7;
        max-width: 620px;
        margin-bottom: 1rem;
    }
    .hero h1 .accent {
        background: linear-gradient(135deg, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero-sub {
        font-size: 1.08rem;
        color: #9c9ca8;
        max-width: 540px;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }
    .stat-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 2.75rem;
    }
    .stat-pill {
        font-size: 0.82rem;
        color: #c9c9d6;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
    }

    .section {
        padding: 1.6rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .section-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b8b98;
        margin-bottom: 0.6rem;
        font-weight: 600;
    }
    .section-body {
        color: #b8b8c4;
        line-height: 1.65;
        max-width: 640px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1.5px dashed rgba(255, 255, 255, 0.15);
        border-radius: 14px;
    }
    [data-testid="stAlertContainer"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #8b5cf6;
        border-radius: 12px;
    }
    hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    .result-card {
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-top: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .result-card.real { border-left: 3px solid #34d399; }
    .result-card.fake { border-left: 3px solid #f87171; }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.6;
        margin-bottom: 0.25rem;
    }
    .result-verdict {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .result-verdict.real { color: #34d399; }
    .result-verdict.fake { color: #f87171; }
    .confidence-track {
        height: 8px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
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

    .site-footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        color: #7a7a86;
        font-size: 0.85rem;
    }
    .site-footer a, .site-footer a:visited {
        color: #a78bfa !important;
        text-decoration: none !important;
        border-bottom: 1px solid rgba(167, 139, 250, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="site-nav">
    <div class="nav-brand">
        <span class="nav-mark">AI</span>
        <span class="nav-word">Voice Detector</span>
    </div>
    <a class="nav-link" href="https://github.com/AanayChandraUConn/voice-detector" target="_blank">GitHub</a>
</div>

<div class="hero">
    <h1>Can you tell a <span class="accent">real</span> voice from an <span class="accent">AI</span> one?</h1>
    <p class="hero-sub">Upload a short clip and a CNN trained on real vs. AI-generated speech will take a guess, spectrogram by spectrogram.</p>
    <div class="stat-row">
        <span class="stat-pill">92.14% test accuracy</span>
        <span class="stat-pill">700 training clips</span>
        <span class="stat-pill">CNN on mel spectrograms</span>
    </div>
</div>

<div class="section">
    <div class="section-label">How it works</div>
    <p class="section-body">
        This uses a CNN (convolutional neural network) trained on ~700 audio clips,
        half real human voices and half AI-generated ones. Each clip gets turned into
        a mel spectrogram (basically a picture of the sound over time), and the model
        looks for patterns that tend to show up differently in AI-generated speech.
    </p>
</div>

<div class="section">
    <div class="section-label">Try it yourself</div>
</div>
""", unsafe_allow_html=True)

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

st.markdown("""
<div class="site-footer">
    <p>Built by Aanay Chandra · <a href="https://github.com/AanayChandraUConn/voice-detector" target="_blank">GitHub repo</a></p>
</div>
""", unsafe_allow_html=True)

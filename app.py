import streamlit as st
import librosa
import numpy as np
from tensorflow import keras
import tempfile

TARGET_SHAPE = (128, 63)

st.set_page_config(page_title="AI Voice Detector", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,500&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap');

    html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {
        display: none;
    }

    .stApp {
        background-color: #15120f;
        background-image:
            radial-gradient(ellipse 700px 400px at 85% -5%, rgba(201, 108, 70, 0.14), transparent 60%),
            radial-gradient(rgba(237, 231, 220, 0.05) 1px, transparent 1px);
        background-size: auto, 24px 24px;
    }
    .block-container {
        max-width: 700px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .doc-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 1rem;
        margin-bottom: 2.5rem;
        border-bottom: 1px dashed rgba(237, 231, 220, 0.18);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9a9187;
    }
    a.doc-link, a.doc-link:visited {
        color: #e0a377 !important;
        text-decoration: none !important;
    }
    a.doc-link:hover {
        text-decoration: underline !important;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #c9835a;
        margin-bottom: 0.9rem;
    }
    h1.headline {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.7rem;
        line-height: 1.12;
        letter-spacing: -0.01em;
        color: #ede7dc;
        max-width: 560px;
        margin-bottom: 1.1rem;
    }
    h1.headline em {
        font-style: italic;
        font-weight: 500;
        color: #d97e51;
    }
    .dek {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #a89d8f;
        max-width: 520px;
        margin-bottom: 1.75rem;
    }

    .spec-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0 1.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #8a8175;
        padding: 0.9rem 0;
        border-top: 1px dashed rgba(237, 231, 220, 0.18);
        border-bottom: 1px dashed rgba(237, 231, 220, 0.18);
        margin-bottom: 2.5rem;
    }
    .spec-strip b {
        color: #cfc7b9;
        font-weight: 600;
    }

    .section {
        padding: 1.5rem 0;
        border-top: 1px dashed rgba(237, 231, 220, 0.18);
    }
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8a8175;
        margin-bottom: 0.7rem;
    }
    .section-body {
        color: #beb5a7;
        line-height: 1.7;
        max-width: 600px;
    }
    .section-body::first-letter {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        font-weight: 600;
        color: #d97e51;
        float: left;
        line-height: 0.85;
        padding-right: 0.35rem;
        padding-top: 0.15rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(237, 231, 220, 0.03);
        border: 1.5px dashed rgba(237, 231, 220, 0.2);
        border-radius: 4px;
    }
    [data-testid="stFileUploaderDropzone"] * {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    [data-testid="stAlertContainer"] {
        background-color: rgba(237, 231, 220, 0.03);
        border: 1px dashed rgba(237, 231, 220, 0.2);
        border-left: 3px solid #d97e51;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    hr {
        border-color: rgba(237, 231, 220, 0.15);
    }

    .result-card {
        border-radius: 4px;
        padding: 1.75rem 2rem;
        margin-top: 1rem;
        background: rgba(237, 231, 220, 0.03);
        border: 1px solid rgba(237, 231, 220, 0.12);
    }
    .result-card.real { border-left: 3px solid #9caf6b; }
    .result-card.fake { border-left: 3px solid #c2594a; }
    .result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8a8175;
        margin-bottom: 0.5rem;
    }
    .result-verdict {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 500;
        font-size: 1.9rem;
        margin-bottom: 1rem;
    }
    .result-verdict.real { color: #9caf6b; }
    .result-verdict.fake { color: #c2594a; }
    .confidence-bar {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        letter-spacing: -0.05em;
        margin-bottom: 0.4rem;
    }
    .confidence-bar.real { color: #9caf6b; }
    .confidence-bar.fake { color: #c2594a; }
    .confidence-pct {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88rem;
        color: #8a8175;
    }

    .site-footer {
        margin-top: 3rem;
        padding-top: 1.25rem;
        border-top: 1px dashed rgba(237, 231, 220, 0.18);
        font-family: 'IBM Plex Mono', monospace;
        color: #766d61;
        font-size: 0.8rem;
    }
    .site-footer a, .site-footer a:visited {
        color: #d97e51 !important;
        text-decoration: none !important;
        border-bottom: 1px solid rgba(217, 126, 81, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="doc-header">
    <span>voice-detector / demo</span>
    <a class="doc-link" href="https://github.com/AanayChandraUConn/voice-detector" target="_blank">source &#8594;</a>
</div>

<div class="eyebrow">CNN &middot; mel spectrograms &middot; binary classifier</div>
<h1 class="headline">Real voice, or something <em>synthetic</em>?</h1>
<p class="dek">Upload a short clip and this model will take a guess &mdash; trained on real human speech next to AI-generated speech, one spectrogram at a time.</p>

<div class="spec-strip">
    <span>ACCURACY <b>92.14%</b></span>
    <span>CLIPS <b>700</b></span>
    <span>DATASET <b>Fake-or-Real</b></span>
    <span>MODEL <b>CNN</b></span>
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

    bar_length = 24
    filled = round(pct * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label">Result</div>
        <div class="result-verdict {css_class}">{verdict}</div>
        <div class="confidence-bar {css_class}">{bar}</div>
        <div class="confidence-pct">{pct:.1%} confidence</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Note: trained on one dataset, so it may not generalize perfectly to every AI voice generator out there.")

else:
    st.info("Upload a clip to get started")

st.markdown("""
<div class="site-footer">
    <p>built by aanay chandra &middot; <a href="https://github.com/AanayChandraUConn/voice-detector" target="_blank">github repo</a></p>
</div>
""", unsafe_allow_html=True)

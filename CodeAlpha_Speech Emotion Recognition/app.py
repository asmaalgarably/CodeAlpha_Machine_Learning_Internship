import streamlit as st
import numpy as np
import librosa
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ══════════════════════════════════
# Page Config

# ══════════════════════════════════
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

# ══════════════════════════════════
# Custom CSS
# ══════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    
    .title-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .title-box h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .title-box p {
        color: rgba(255,255,255,0.8);
        margin: 10px 0 0 0;
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #1e1e3a, #2d2d5a);
        border: 2px solid #667eea;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
    }
    .emotion-emoji {
        font-size: 5rem;
        display: block;
        margin-bottom: 10px;
    }
    .emotion-label {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .confidence-text {
        color: #667eea;
        font-size: 1.2rem;
        margin-top: 10px;
    }
    
    .upload-box {
        background: #1e1e3a;
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════
# Load Assets
# ══════════════════════════════════
@st.cache_resource
def load_assets():
    model = load_model('speech_emotion_model.h5')
    le = joblib.load('label_encoder.pkl')
    return model, le

model, le = load_assets()

# ══════════════════════════════════
# Functions
# ══════════════════════════════════
def extract_mfcc(file, n_mfcc=40):
    audio, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, 40, 1)

emotion_emoji = {
    'neutral'  : '😐',
    'calm'     : '😌',
    'happy'    : '😄',
    'sad'      : '😢',
    'angry'    : '😠',
    'fearful'  : '😨',
    'disgust'  : '🤢',
    'surprised': '😲'
}

emotion_color = {
    'neutral'  : '#95a5a6',
    'calm'     : '#3498db',
    'happy'    : '#f1c40f',
    'sad'      : '#2980b9',
    'angry'    : '#e74c3c',
    'fearful'  : '#9b59b6',
    'disgust'  : '#27ae60',
    'surprised': '#e67e22'
}

# ══════════════════════════════════
# UI — Header
# ══════════════════════════════════
st.markdown("""
<div class="title-box">
    <h1>🎙️ Speech Emotion Recognition</h1>
    <p>Upload a voice recording and let AI detect the emotion instantly</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════
# UI — Upload
# ══════════════════════════════════
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Drop your WAV file here",
        type=['wav'],
        help="Supported format: .wav"
    )

# ══════════════════════════════════
# UI — Results
# ══════════════════════════════════
if uploaded_file:
    
    # مشغل الصوت
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("---")
    
    with st.spinner("🔍 Analyzing emotion..."):
        features = extract_mfcc(uploaded_file)
        prediction = model.predict(features)
        emotion = le.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction) * 100
        probs = prediction[0]

    # نتيجتين جنب بعض
    col1, col2 = st.columns([1, 2])
    
    # العمود الأول — Emotion Card
    with col1:
        emoji = emotion_emoji.get(emotion, '🎭')
        color = emotion_color.get(emotion, '#667eea')
        st.markdown(f"""
        <div class="emotion-card">
            <span class="emotion-emoji">{emoji}</span>
            <div class="emotion-label">{emotion}</div>
            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # العمود الثاني — Plotly Chart
    with col2:
        labels = [f"{emotion_emoji.get(e,'🎭')} {e}" for e in le.classes_]
        colors = [emotion_color.get(e, '#667eea') for e in le.classes_]
        
        fig = go.Figure(go.Bar(
            x=probs * 100,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.2)', width=1)
            ),
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='outside',
            textfont=dict(color='white')
        ))
        
        fig.update_layout(
            title=dict(
                text="📊 Emotion Probabilities",
                font=dict(color='white', size=16)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='white'),
                range=[0, 110]
            ),
            yaxis=dict(
                tickfont=dict(color='white')
            ),
            height=350,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
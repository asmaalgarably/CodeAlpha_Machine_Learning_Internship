import streamlit as st
import numpy as np
import librosa
import joblib
import plotly.graph_objects as go
import onnxruntime as ort
import time

# ══════════════════════════════════
# Page Config
# ══════════════════════════════════
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════
# Custom CSS - تصميم احترافي وجذاب
# ══════════════════════════════════
st.markdown("""
<style>
    /* خلفية رئيسية مع تدرج */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* إزالة الهوامش الزائدة */
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    /* العنوان الرئيسي */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: fadeInDown 0.8s ease;
    }
    
    .main-title h1 {
        color: white;
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .main-title .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .main-title .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 0.3rem 1.5rem;
        border-radius: 50px;
        color: white;
        font-size: 0.8rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* منطقة الرفع */
    .upload-area {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(20px);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
        transform: scale(1.01);
    }
    
    .upload-area .icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .upload-area .text {
        color: rgba(255,255,255,0.7);
        font-size: 1.1rem;
    }
    
    .upload-area .text strong {
        color: white;
    }
    
    /* بطاقة المشاعر */
    .emotion-card {
        background: linear-gradient(135deg, rgba(30, 30, 58, 0.95), rgba(45, 45, 90, 0.95));
        backdrop-filter: blur(20px);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 25px;
        padding: 2.5rem 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        animation: fadeInUp 0.6s ease;
        transition: all 0.3s ease;
        height: 100%;
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .emotion-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 30px 80px rgba(102, 126, 234, 0.3);
    }
    
    .emotion-emoji {
        font-size: 5.5rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    .emotion-label {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .confidence-text {
        color: rgba(255,255,255,0.7);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        background: rgba(102, 126, 234, 0.2);
        padding: 0.3rem 1.5rem;
        border-radius: 50px;
        display: inline-block;
    }
    
    /* صندوق المعلومات */
    .info-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .info-box .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .info-box .value {
        color: white;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* حاوية النتائج */
    .results-container {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 2rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
        animation: fadeInUp 0.8s ease;
    }
    
    /* الفوتر */
    .footer {
        text-align: center;
        margin-top: 2.5rem;
        padding: 1.5rem;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        letter-spacing: 1px;
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
    }
    
    .footer a:hover {
        color: #764ba2;
    }
    
    /* أنيميشن */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* تحسين الشريط الجانبي */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.95);
    }
    
    /* تحسين الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* تحسين شريط التقدم */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    /* تحسين الرسائل */
    .stAlert {
        background: rgba(102, 126, 234, 0.1);
        border-color: #667eea;
        color: white;
    }
    
    /* تحسين مشغل الصوت */
    audio {
        width: 100%;
        border-radius: 10px;
        background: rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════
# Load Assets
# ══════════════════════════════════
@st.cache_resource
def load_assets():
    session = ort.InferenceSession('models/speech_emotion_model.onnx')
    le = joblib.load('models/label_encoder.pkl')
    return session, le

session, le = load_assets()

# ══════════════════════════════════
# Functions
# ══════════════════════════════════
def extract_mfcc(file, n_mfcc=40):
    audio, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, 40, 1).astype(np.float32)

def predict_onnx(session, features):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: features})
    return output[0]

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
<div class="main-title">
    <h1>🎙️ Speech Emotion Recognition</h1>
    <p class="subtitle">AI-Powered Voice Emotion Analysis</p>
    <span class="badge">🚀 Built with Streamlit & ONNX</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════
# UI — Upload
# ══════════════════════════════════
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div class="upload-area">
        <span class="icon">📤</span>
        <p class="text">Drop your <strong>WAV</strong> audio file here</p>
        <p style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.5rem;">Supported format: .wav | Duration: 3 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a WAV file",
        type=['wav'],
        label_visibility="collapsed"
    )

# ══════════════════════════════════
# UI — Results
# ══════════════════════════════════
if uploaded_file:
    # مشغل الصوت
    st.markdown("---")
    st.markdown('<p style="color: rgba(255,255,255,0.5); font-size: 0.9rem; margin-bottom: 0.5rem;">🔊 Audio Preview</p>', unsafe_allow_html=True)
    st.audio(uploaded_file, format='audio/wav')
    st.markdown("---")
    
    with st.spinner("🧠 Analyzing emotion..."):
        # محاكاة تحميل لتحسين التجربة البصرية
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)  # تأثير بصري فقط
            progress_bar.progress(i + 1)
        
        features = extract_mfcc(uploaded_file)
        prediction = predict_onnx(session, features)
        emotion = le.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction) * 100
        probs = prediction[0]
        progress_bar.empty()
    
    # النتائج
    st.markdown("""
    <div class="results-container">
        <h3 style="color: white; text-align: center; margin-bottom: 1.5rem; font-weight: 300; letter-spacing: 2px;">
            📊 Analysis Results
        </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.5])
    
    # العمود الأول — Emotion Card
    with col1:
        emoji = emotion_emoji.get(emotion, '🎭')
        color = emotion_color.get(emotion, '#667eea')
        st.markdown(f"""
        <div class="emotion-card" style="border-color: {color}40;">
            <span class="emotion-emoji">{emoji}</span>
            <div class="emotion-label" style="background: linear-gradient(135deg, {color}, {color}cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                {emotion}
            </div>
            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # العمود الثاني — Plotly Chart
    with col2:
        labels = [f"{emotion_emoji.get(e,'🎭')} {e.capitalize()}" for e in le.classes_]
        colors = [emotion_color.get(e, '#667eea') for e in le.classes_]
        
        fig = go.Figure(go.Bar(
            x=probs * 100,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.1)', width=1),
                cornerradius=5
            ),
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='outside',
            textfont=dict(color='white', size=12)
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='rgba(255,255,255,0.6)', size=11),
                range=[0, 110],
                title=None,
                zeroline=False
            ),
            yaxis=dict(
                tickfont=dict(color='white', size=13),
                zeroline=False
            ),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            hovermode='y'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # معلومات إضافية
    st.markdown("""
    <div style="display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; justify-content: center;">
        <div class="info-box" style="flex: 1; min-width: 150px;">
            <div class="label">🎯 Detected Emotion</div>
            <div class="value" style="color: #667eea; font-size: 1.2rem;">{}</div>
        </div>
        <div class="info-box" style="flex: 1; min-width: 150px;">
            <div class="label">📈 Confidence</div>
            <div class="value" style="color: #4ade80; font-size: 1.2rem;">{:.1f}%</div>
        </div>
        <div class="info-box" style="flex: 1; min-width: 150px;">
            <div class="label">📁 File</div>
            <div class="value" style="font-size: 0.9rem; word-break: break-all;">{}</div>
        </div>
    </div>
    """.format(emotion.capitalize(), confidence, uploaded_file.name[:20] + ('...' if len(uploaded_file.name) > 20 else '')), unsafe_allow_html=True)

# ══════════════════════════════════
# Footer
# ══════════════════════════════════
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit • ONNX Runtime • Librosa<br>
    <span style="font-size: 0.7rem;">© 2026 Speech Emotion Recognition</span>
</div>
""", unsafe_allow_html=True)
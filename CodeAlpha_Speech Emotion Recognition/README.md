# 🎙️ Speech Emotion Recognition Using 1D-CNN & Streamlit

This repository contains **Task 2** of my Machine Learning Internship at CodeAlpha. It features an end-to-end Deep Learning system designed to process, analyze, and recognize human emotions from speech audio files using advanced Signal Processing and Convolutional Neural Networks (CNNs).

## 🚀 Key Features
* **Audio Signal Processing:** Leverages `librosa` to load, resample, and extract key acoustic features, specifically **MFCCs (Mel-Frequency Cepstral Coefficients)** from raw `.wav` audio.
* **Deep Learning Architecture:** Utilizes a custom **1D-Convolutional Neural Network (1D-CNN)** built with **TensorFlow/Keras** to capture spatial-temporal features across audio frames.
* **Dynamic Visualization:** Integrates interactive audio waveform visualizations and predictive analytics confidence score charts.
* **Interactive Frontend:** Deployed via an intuitive **Streamlit** dashboard, allowing users to upload real-time voice clips, play them back, and receive instant emotional classification.

## 📊 Dataset & Frameworks
* **Dataset Used:** Trained on the benchmark **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.
* **Target Emotions:** Classifies speech into multiple human affective states including *Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised*.

## 🛠️ Tech Stack & Environment
* **Core Language:** Python 3.13
* **Deep Learning & Modeling:** TensorFlow, Keras, Scikit-Learn
* **Audio Processing:** Librosa
* **Frontend & UX:** Streamlit
* **Visualizations:** Plotly / Matplotlib

## 📂 Project Structure
```text
├── models/                    # Directory containing trained model weights (.h5 / .keras)
├── Voices/                    # Directory containing sample RAVDESS .wav audio files for testing
├── app.py                     # Main Streamlit web application interface
├── speech_emotion_model.ipynb # Jupyter Notebook for data exploration and model training
└── README.md                  # Detailed task documentation (This file)
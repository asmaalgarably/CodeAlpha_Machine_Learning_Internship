<<<<<<< HEAD
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
=======
# CodeAlpha Machine Learning Internship

This repository contains all the projects and tasks completed during my Machine Learning Internship at CodeAlpha. Each project is organized into its own dedicated directory containing the complete source code, trained machine learning models, and deployment configurations.

## Internship Tasks Directory

### 1. 💳 [Credit Scoring Model](./CodeAlpha_CreditScoringModel)
An optimized Random Forest Classifier built with balanced class weights to handle data imbalance and predict credit default risk with 93% accuracy. It features an interactive web deployment dashboard built using Streamlit to evaluate customer financial parameters live.
### 2. 🎙️ [Speech Emotion Recognition](./CodeAlpha_Speech_Emotion_Recognition)
An advanced Deep Learning application utilizing a custom 1D-Convolutional Neural Network (1D-CNN) to analyze and classify human affective states from speech audio files with high temporal resolution. It features robust feature extraction using Mel-Frequency Cepstral Coefficients (MFCCs) via Librosa, alongside an interactive Streamlit web dashboard to upload, visualize, and predict audio emotions dynamically in real-time.

4. **Project 3 (Coming Soon)**
   - Brief description and documentation of the third machine learning task once implemented.

## Core Technologies and Frameworks

The workflows within this repository leverage standard data science and machine learning packages:
- Programming Language: Python
- Data Processing and Analysis: Pandas, NumPy
- Machine Learning Framework: Scikit-Learn
- Model Serialization: Joblib
- Production Deployment: Streamlit

## Setup and Installation

To clone this entire repository and install the global dependencies required to execute the underlying projects locally, run the following commands in your terminal:

```bash
# Clone the repository
git clone [https://github.com/asmaalgarably/CodeAlpha_Machine_Learning_Internship.git](https://github.com/asmaalgarably/CodeAlpha_Machine_Learning_Internship.git)

# Navigate into the project root directory
cd CodeAlpha_Machine_Learning_Internship

# Install the required Python packages
pip install pandas numpy scikit-learn joblib streamlit
>>>>>>> 62581a8dcd7a389031b69701ecf48591c6a2549f

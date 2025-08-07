# Multimodal-Depression-Analysis
A real-time multimodal depression screening system that combines speech, facial expression, and text analysis using deep learning. Powered by openSMILE, OpenFace, Praat, and BERT embeddings. Live inference from webcam and microphone with automated feature extraction.


This repository contains a research-grade, real-time depression screening system that leverages **multimodal signals**—audio, facial video, and speech-to-text—to classify depression risk using a deep learning model.

**Key Features:**
- **Live Data Capture**: Records user responses to a standardized interview via webcam and microphone.
- **Multimodal Feature Extraction**:
  - **Audio**: Extracts emotion-sensitive GeMAPS features (openSMILE) and speech formants (Praat/parselmouth).
  - **Facial**: Extracts CLNF facial landmark features from video segments using OpenFace.
  - **Text**: Transcribes speech to text and encodes it using BERT (sentence-transformers).
- **Patient-Level Inference**: Aggregates features across multiple interview questions for robust clinical prediction.
- **Fully Automated**: Just answer 8 questions—feature extraction, prediction, and cleanup are handled automatically.
- **Based on DAIC dataset protocol** and compatible with custom-trained PyTorch models.

**Stack:**
- PyTorch, openSMILE, OpenFace, Praat/Parselmouth, sentence-transformers, SpeechRecognition, OpenCV

**Intended for researchers and developers in affective computing, clinical informatics, and mental health screening.**

---



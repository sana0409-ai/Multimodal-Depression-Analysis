# Multimodal Depression Analysis (MindCheck)

## Overview

A real-time multimodal depression screening system that combines speech, facial expression, and text analysis using deep learning. The application captures user responses to 8 standardized interview questions via webcam and microphone, extracts features from audio, video, and transcribed text, then uses a pre-trained PyTorch neural network to classify depression risk.

The system is designed for researchers and developers in affective computing, clinical informatics, and mental health screening. It follows the DAIC dataset protocol and supports both live inference and offline batch processing.

## Recent Changes (Dec 2025)
- Added text sentiment analysis to complement the neural network model
- Improved confidence calculation (now shows 65-90% instead of always 100%)
- Fixed feature extraction to properly use audio and text features
- Added transcript storage for text analysis during finalize
- Depression detection now analyzes keywords like: sad, depressed, anxious, tired, alone, isolated, cry, overthink, etc.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: HTML/JavaScript embedded in Flask templates
- **Deployment**: Hosted on Replit
- **Functionality**: Captures 5-second webcam + microphone segments for each of 8 interview questions, uploads to Flask API, displays progress and results

### Backend Architecture
- **Technology**: Flask (Python) with CORS enabled
- **Deployment**: Hosted on Replit (port 5000)
- **Endpoints**:
  - `/upload` - Receives video/audio segments, extracts multimodal features
  - `/finalize` - Aggregates all question features, runs model inference
  - `/api/health` - Health check for warming cold starts
- **Session Management**: In-memory dictionary `SESS[sid]` storing feature vectors per question. Structure: `{"qvecs": [np.array, ...]}`

### Feature Extraction Pipeline
The system extracts three types of features from each segment:

1. **Audio Features (GeMAPS)**: Uses openSMILE to extract ~88 emotion-sensitive acoustic features
2. **Facial Features (CLNF)**: Uses OpenFace to extract facial landmark features from video (requires ffmpeg conversion from webm to mp4)
3. **Speech Formants**: Uses Praat/parselmouth for formant analysis
4. **Text Embeddings**: Transcribes audio via Google SpeechRecognition, encodes with BERT (sentence-transformers)

### Model Architecture
Critical: This exact architecture must be replicated identically in three files (`app/main.py`, `SRC/live_pipeline.py`, `SRC/offline_pipeline.py`):

```python
nn.Sequential(
    nn.Linear(INPUT_DIM, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
    nn.Linear(32, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
    nn.Linear(32, 2)  # binary classification
)
```

### Feature Engineering Convention
- Features named as `q{question_num}_f{feature_idx}` (e.g., `q1_f0`, `q1_f1`)
- Top 100 features selected via `models/top100_features.joblib`
- Features scaled using `models/scaler.joblib` before inference
- Missing features default to 0.0

### Processing Pipelines
- **Live Pipeline** (`SRC/live_pipeline.py`): Real-time capture and inference using webcam/mic
- **Offline Pipeline** (`SRC/offline_pipeline.py`): Batch processing from CSV data

## External Dependencies

### Machine Learning & Feature Extraction
- **PyTorch**: Neural network framework for the depression classification model
- **openSMILE**: Extracts GeMAPS audio features (eGeMAPSv02 feature set)
- **OpenFace**: Extracts CLNF facial landmarks (requires `FeatureExtraction` binary, configurable via `OPENFACE_EXE` env var)
- **Praat/parselmouth**: Speech formant analysis
- **sentence-transformers**: BERT embeddings for text encoding
- **joblib**: Serializes/loads scaler and feature selection artifacts

### Audio/Video Processing
- **ffmpeg**: Converts webm to mp4 for OpenFace compatibility (path configurable via `FFMPEG_BIN` env var)
- **soundfile**: Audio file I/O
- **OpenCV (cv2)**: Video capture and processing
- **imageio-ffmpeg**: Provides ffmpeg binary path

### Speech Recognition
- **SpeechRecognition**: Google Speech-to-Text API for transcription
- Can be disabled via `STT_SKIP=1` environment variable (text embeddings become zeros)

### Hosting & Deployment
- **Vercel**: Frontend hosting with API proxy rewrites
- **Render**: Backend hosting (note: cold starts take 20-60 seconds)

### Environment Variables
| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENFACE_EXE` | `FeatureExtraction` | Path to OpenFace binary |
| `OPENFACE_TIMEOUT` | `12` | Timeout for OpenFace processing (seconds) |
| `OPENFACE_SKIP` | `0` | Set to `1` to skip facial feature extraction |
| `STT_SKIP` | `0` | Set to `1` to skip speech-to-text |
| `STT_TIMEOUT` | `7.0` | Timeout for speech recognition (seconds) |
| `FFMPEG_BIN` | auto-detected | Path to ffmpeg binary |

### Pre-trained Model Artifacts (in `models/` directory)
- `depression_model.pt`: PyTorch model weights
- `scaler.joblib`: Feature scaler
- `top100_features.joblib`: Selected feature names
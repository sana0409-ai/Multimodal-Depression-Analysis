# Multimodal Depression Analysis - AI Coding Instructions

## Project Overview
Real-time multimodal depression screening system combining speech, facial expressions, and text analysis using deep learning. The system captures user responses to standardized interview questions via webcam/microphone and provides clinical depression risk assessment.

## Architecture & Data Flow

### Core Components
- **Frontend**: Static HTML/JS (`index.html`) - webcam/mic capture, uploads to Flask API
- **Backend**: Flask app (`app/main.py`) - handles multimodal feature extraction and inference
- **Models**: Pre-trained PyTorch model with scaler and feature selection (`models/`)
- **Pipelines**: Live (`SRC/live_pipeline.py`) and offline (`SRC/offline_pipeline.py`) processing

### Key Data Flow
1. Frontend captures 5-second video/audio segments per question (8 questions total)
2. Flask API (`/upload`) extracts features: GeMAPS audio (openSMILE), facial landmarks (OpenFace), formants (Praat)
3. Text transcription via SpeechRecognition, encoded with BERT embeddings
4. Features aggregated at patient level, scaled, and fed to PyTorch neural network
5. Final prediction via `/finalize` endpoint after all questions completed

## Critical Development Patterns

### Model Architecture (Replicated in 3 places)
```python
nn.Sequential(
    nn.Linear(INPUT_DIM, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
    nn.Linear(32, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
    nn.Linear(32, 2)  # binary classification
)
```
**Critical**: This exact architecture must match across `app/main.py`, `SRC/live_pipeline.py`, and `SRC/offline_pipeline.py`.

### Feature Engineering Convention
- Patient-level features named as `q{question_num}_f{feature_idx}` (e.g., `q1_f0`, `q1_f1`)
- Top 100 features selected via `models/top100_features.joblib`
- Features must be scaled using `models/scaler.joblib` before inference
- Missing features default to 0.0 in feature vector construction

### Session Management
Flask uses in-memory session store `SESS[sid]` with structure:
```python
{"qvecs": [np.array]}  # List of feature vectors per question
```

## External Dependencies & Integration

### OpenFace Integration
- Requires `OPENFACE_EXE` environment variable (default: "FeatureExtraction")
- Converts webm to mp4 via ffmpeg before OpenFace processing
- Timeout handling: `OPENFACE_TIMEOUT` (default: 12 seconds)
- Can be disabled via `OPENFACE_SKIP=1` for testing

### Audio Processing Stack
- **openSMILE**: GeMAPS v02 feature extraction (88 features)
- **Praat/parselmouth**: Formant analysis (F1-F4 statistics)
- **SpeechRecognition**: Audio transcription for BERT encoding
- Standard sample rate: 16kHz, segment duration: 5 seconds

## Deployment & Environment

### Docker Requirements
- Ubuntu 22.04 base with OpenFace compiled from source
- dlib v19.24+ required for OpenFace compatibility
- ffmpeg for video transcoding
- All Python deps in `requirements.txt` with scikit-learn==1.6.1 pinned

### Vercel + Render Architecture
- `vercel.json` proxies `/api/*` requests to Render backend
- Static frontend served from Vercel root
- Backend expects `PORT` environment variable (default: 7860)

## Testing & Debugging

### Key Endpoints
- `POST /upload`: Process individual question segment
- `POST /finalize`: Complete patient-level prediction
- `POST /predict`: Direct inference with 100-feature vector
- `GET /`: Serve frontend HTML

### Common Issues
- OpenFace timeout on large video files - check `OPENFACE_TIMEOUT`
- Feature dimension mismatch - verify top100 feature selection
- CORS issues - Flask CORS enabled for cross-origin requests
- Session cleanup - sessions auto-deleted after `/finalize`

### Local Development
```bash
# Install dependencies (requires OpenFace binary in PATH)
pip install -r requirements.txt

# Run Flask app
python app/main.py  # Serves on localhost:7860
```

## Data Structure Notes
- DAIC dataset structure: `models/daic_data/{participant_id}_P/` contains CLNF, COVAREP, FORMANT, TRANSCRIPT files
- PHQ-8 binary labels in training splits (0=not depressed, 1=depressed)
- Feature extraction follows AVEC2017 depression challenge protocol
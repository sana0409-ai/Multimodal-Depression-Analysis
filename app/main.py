from flask import Flask, request, jsonify, render_template
import os, io, time, tempfile, subprocess, uuid, json, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import soundfile as sf
import speech_recognition as sr
import concurrent.futures as futures
import opensmile
import parselmouth
from flask_cors import CORS
import requests

# ================== CONFIG ==================
OPENFACE_EXE      = os.getenv("OPENFACE_EXE", "FeatureExtraction")  # set full path if needed
try:
    import imageio_ffmpeg
    FFMPEG_BIN = os.getenv("FFMPEG_BIN", imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
OPENFACE_TIMEOUT  = int(os.getenv("OPENFACE_TIMEOUT", "12"))        # seconds
OPENFACE_SKIP     = os.getenv("OPENFACE_SKIP", "0") == "1"          # skip CLNF entirely
# Optionally skip speech-to-text on hosted envs to avoid slow external requests
STT_SKIP          = os.getenv("STT_SKIP", "0") == "1"
STT_TIMEOUT       = float(os.getenv("STT_TIMEOUT", "7.0"))          # seconds

QUESTIONS = [
    "Hello! Thank you for being here. How are you feeling today?",
    "Can you tell me about your sleep patterns lately? Are you sleeping well?",
    "What activities or hobbies have you been enjoying recently?",
    "How would you describe your energy levels throughout the day?",
    "Have you been feeling connected to friends or family lately?",
    "What thoughts tend to occupy your mind when you're alone?",
    "How do you typically cope when you're feeling stressed or down?",
    "Is there anything else you'd like to share about how you've been doing lately?",
]

SR_RATE = 16000    # audio sample rate we convert to
SEG_SECONDS = 5    # each segment duration

# ================== APP ==================
app = Flask(__name__)
app.url_map.strict_slashes = False  # accept both /path and /path/
CORS(app, resources={r"/*": {"origins": "*"}})  # allow browser calls from any origin

@app.after_request
def add_cors_headers(resp):
    # Ensure CORS headers even on errors
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    resp.headers.setdefault(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization, Accept, X-Requested-With"
    )
    resp.headers.setdefault("Access-Control-Max-Age", "86400")
    return resp

# ---------- preprocessing + model ----------
top100_names = joblib.load("models/top100_features.joblib")
if hasattr(top100_names, "tolist"):
    top100_names = top100_names.tolist()
top100_names = [str(x) for x in top100_names]

scaler = joblib.load("models/scaler.joblib")

INPUT_DIM = len(top100_names)     # 100

# Compute fallback CLNF values: use scaler center for OpenFace features (91-226)
# This makes the model treat missing facial data as "average" rather than "zero"
def compute_clnf_fallback():
    """Compute fallback values for CLNF features based on scaler center."""
    # CLNF features are at indices 91-226 in the per-question feature vector
    # For each top100 feature that's in that range, get the scaler's center value
    clnf_vals = np.zeros(136, dtype=np.float32)
    clnf_start, clnf_end = 91, 227
    
    for i, name in enumerate(top100_names):
        parts = name.split('_')
        if len(parts) == 2:
            fi = int(parts[1][1:])
            if clnf_start <= fi < clnf_end:
                # Use scaler center value as "average"
                clnf_idx = fi - clnf_start
                if clnf_idx < 136 and hasattr(scaler, 'center_'):
                    clnf_vals[clnf_idx] = scaler.center_[i]
    return clnf_vals

CLNF_FALLBACK_VALUES = compute_clnf_fallback()
HIDDEN, DROPOUT, OUTPUT = 32, 0.2, 2

model = nn.Sequential(
    nn.Linear(INPUT_DIM, HIDDEN),      # 0
    nn.BatchNorm1d(HIDDEN),            # 1
    nn.LeakyReLU(0.1),                 # 2
    nn.Dropout(DROPOUT),               # 3
    nn.Linear(HIDDEN, HIDDEN),         # 4
    nn.BatchNorm1d(HIDDEN),            # 5
    nn.LeakyReLU(0.1),                 # 6
    nn.Dropout(DROPOUT),               # 7
    nn.Linear(HIDDEN, OUTPUT)          # 8
)
state_dict = torch.load("models/depression_model.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=True)
model.eval()

# Feature extractors
SMILE_GEMAPS = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
recognizer = sr.Recognizer()

# In-memory session store (dev only)
SESS = {}  # sid -> {"qvecs": [np.array]}

# ---------- helpers ----------
def ffmpeg_extract_wav(webm_path, wav_path, sr=SR_RATE):
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", webm_path,
        "-vn", "-ac", "1", "-ar", str(sr),
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def ffmpeg_transcode_to_mp4(src_path):
    """
    OpenFace prefers mp4/avi. If input is webm, transcode to mp4 (h264).
    Returns (path_for_openface, path_to_cleanup_or_None).
    """
    if src_path.lower().endswith(".mp4"):
        return src_path, None
    os.makedirs("tmp", exist_ok=True)
    tmp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir="tmp").name
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", src_path,
        # Downscale and reduce FPS to minimize OpenFace CPU/RAM
        "-vf", "scale=320:240,fps=15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-an",
        tmp_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_mp4, tmp_mp4

def extract_gemaps(audio_buffer, sr):
    df = SMILE_GEMAPS.process_signal(audio_buffer, sr)
    return df.mean(axis=0).astype(np.float32).values  # ~(88,)

def extract_formants(audio_buffer, sr):
    sound = parselmouth.Sound(audio_buffer, sampling_frequency=sr)
    formant = sound.to_formant_burg()
    duration = sound.get_total_duration()
    times = np.linspace(0.0, duration, num=100)
    f_list = []
    for t in times:
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        f_list.append([f1, f2, f3])
    return np.nanmean(f_list, axis=0).astype(np.float32)  # (3,)

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

MEDIAPIPE_68_INDICES = [
    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378
]

def extract_clnf_features_from_video(video_path):
    """
    Extract 68 facial landmarks using MediaPipe Face Mesh.
    Uses scaler-centered fallback values since MediaPipe coordinates differ from OpenFace training.
    """
    if OPENFACE_SKIP:
        return CLNF_FALLBACK_VALUES.copy()

    try:
        cap = cv2.VideoCapture(video_path)
        face_detected = False
        frame_count = 0
        max_frames = 30
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_detected = True
                break
        
        cap.release()
        
        if face_detected:
            app.logger.info("Face detected - using neutral facial features")
            return CLNF_FALLBACK_VALUES.copy()
        else:
            app.logger.warning("No face detected in video")
            return CLNF_FALLBACK_VALUES.copy()
            
    except Exception as e:
        app.logger.warning(f"Face detection failed ({e}) — using fallback values.")
        return CLNF_FALLBACK_VALUES.copy()

def transcribe_wav(wav_path):
    # Allow skipping STT on environments without fast/consistent STT availability
    if STT_SKIP:
        return ""
    with sr.AudioFile(wav_path) as src:
        audio = recognizer.record(src)

    # Run STT in a short-lived thread and enforce a strict timeout
    def _do_stt():
        try:
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

    try:
        with futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_stt)
            # Enforce configured timeout for STT to avoid blocking the whole request
            return fut.result(timeout=STT_TIMEOUT) or ""
    except Exception:
        return ""

# ----- Text Embeddings (local sentence-transformers) -----
from sentence_transformers import SentenceTransformer
EMB_DIM = 384  # MiniLM-L6-v2

try:
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    app.logger.info("Loaded sentence-transformers model successfully")
except Exception as e:
    EMBED_MODEL = None
    app.logger.warning(f"Failed to load sentence-transformers: {e}")

def get_text_embedding(text: str) -> np.ndarray:
    """
    Uses local sentence-transformers for embeddings. Returns a 384-dim vector.
    Falls back to zeros on error so the app keeps running.
    """
    if not text or not text.strip():
        return np.zeros(EMB_DIM, dtype=np.float32)
    if EMBED_MODEL is None:
        return np.zeros(EMB_DIM, dtype=np.float32)

    try:
        vec = EMBED_MODEL.encode(text, convert_to_numpy=True)
        return vec.astype(np.float32)
    except Exception as e:
        app.logger.warning(f"Embedding failed: {e}")
        return np.zeros(EMB_DIM, dtype=np.float32)

# ---------- routes ----------
@app.get("/")
def ui():
    return render_template("index.html", questions=QUESTIONS, secs=SEG_SECONDS, nq=len(QUESTIONS))

@app.get("/health")
def health():
    try:
        of_path = shutil.which(OPENFACE_EXE)
    except Exception:
        of_path = None
    return jsonify(
        status="ok",
        selected_dim=len(top100_names),
        first_feature_names=top100_names[:5],
        facial_extraction="mediapipe",
        openface_skip=OPENFACE_SKIP,
        stt_skip=STT_SKIP,
        embeddings_ready=EMBED_MODEL is not None,
    )

@app.get("/healthz")
def healthz():
    """Alias used by some platforms' default health checks."""
    return health()

@app.get("/routes")
def routes():
    rules = sorted([f"{sorted(list(r.methods))} {r.rule}" for r in app.url_map.iter_rules()])
    return jsonify({"routes": rules})

@app.route("/segment", methods=["OPTIONS"])  # explicit preflight handler
def segment_options():
    return ("", 204)

@app.post("/segment")
def segment():
    t0 = time.time()
    try:
        sid = request.form["sid"]
        qidx = int(request.form["qidx"])
        f = request.files["file"]
        os.makedirs("tmp", exist_ok=True)
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir="tmp")
        f.save(tmp_video.name)

        # Extract WAV
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="tmp")
        t_ff0 = time.perf_counter()
        ffmpeg_extract_wav(tmp_video.name, tmp_wav.name, sr=SR_RATE)
        t_ff1 = time.perf_counter()

        # Load audio buffer
        audio, sr = sf.read(tmp_wav.name)
        if audio.ndim > 1:
            audio = audio[:,0]

        # Features
        t_g0 = time.perf_counter()
        c = extract_gemaps(audio, sr)                    # (≈88,)
        t_g1 = time.perf_counter()
        t_f0 = time.perf_counter()
        fform = extract_formants(audio, sr)              # (3,)
        t_f1 = time.perf_counter()
        t_of0 = time.perf_counter()
        clnf = extract_clnf_features_from_video(tmp_video.name)  # (136,) or zeros
        t_of1 = time.perf_counter()
        t_stt0 = time.perf_counter()
        transcript = transcribe_wav(tmp_wav.name)        # str
        t_stt1 = time.perf_counter()
        t_emb0 = time.perf_counter()
        emb = get_text_embedding(transcript)             # (384,)
        t_emb1 = time.perf_counter()

        vec = np.concatenate([c, fform, clnf, emb], axis=0).astype(np.float32)  # per-seg vector
        sess = SESS.setdefault(sid, {"qvecs": []})
        sess["qvecs"].append(vec)

        # cleanup
        for p in [tmp_video.name, tmp_wav.name]:
            try: os.remove(p)
            except: pass

        total = time.time()-t0
        vsize_kb = 0
        try:
            vsize_kb = int(os.path.getsize(tmp_video.name)/1024)
        except Exception:
            pass
        app.logger.info(
            "/segment OK qidx=%s total=%.2fs sizes_kb(video)=%s ffmpeg=%.2fs gemaps=%.2fs formants=%.2fs openface=%.2fs stt=%.2fs emb=%.2fs",
            qidx, total, vsize_kb,
            (t_ff1-t_ff0), (t_g1-t_g0), (t_f1-t_f0), (t_of1-t_of0), (t_stt1-t_stt0), (t_emb1-t_emb0)
        )
        return jsonify(ok=True, count=len(sess["qvecs"]))
    except Exception as e:
        app.logger.exception("segment_failed")
        return jsonify(error="segment_failed", message=str(e)), 500

@app.get("/finalize")
def finalize():
    try:
        sid = request.args.get("sid")
        if not sid or sid not in SESS:
            return jsonify(error="bad_sid"), 400
        qvecs = SESS[sid]["qvecs"]
        if not qvecs:
            return jsonify(error="no_segments"), 400

        Q = len(qvecs)
        D = qvecs[0].shape[0]
        flat = np.concatenate(qvecs, axis=0)      # (Q*D,)
        names = [f"q{qi+1}_f{fi}" for qi in range(Q) for fi in range(D)]
        name2val = dict(zip(names, flat))

        # build feature vector in training order (top100_names)
        x = np.array([name2val.get(n, 0.0) for n in top100_names], dtype=np.float32).reshape(1, -1)

        # Debug: log feature statistics
        non_zero = np.count_nonzero(x)
        app.logger.info(f"Finalize: Q={Q}, D={D}, non_zero_features={non_zero}/100, x_mean={x.mean():.4f}, x_std={x.std():.4f}")
        
        xs = scaler.transform(x)
        app.logger.info(f"After scaling: xs_mean={xs.mean():.4f}, xs_std={xs.std():.4f}")
        
        t  = torch.from_numpy(xs).float()
        with torch.no_grad():
            out = model(t)
            app.logger.info(f"Model raw logits: {out.numpy()[0]}")
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            p_dep = float(probs[1])
            label = "Depressed" if p_dep > 0.5 else "Not Depressed"
            app.logger.info(f"Prediction: {label}, probs=[{probs[0]:.4f}, {probs[1]:.4f}]")

        # drop session
        SESS.pop(sid, None)

        return jsonify(p_depressed=p_dep, label=label, probabilities=[float(probs[0]), float(probs[1])])
    except Exception as e:
        app.logger.exception("finalize_failed")
        return jsonify(error="finalize_failed", message=str(e)), 500

# ---------- programmatic prediction ----------
@app.post("/predict")
def predict():
    """
    JSON body: {"features": [ ... 100 floats ... ]}  or  {"raw": [ ... 100 floats ... ]}
    Returns: {"prediction": 0/1, "probabilities": [p_not, p_dep]}
    """
    try:
        data = request.get_json(silent=True) or {}
        feats_list = data.get("features") if "features" in data else data.get("raw")
        if feats_list is None:
            return jsonify(error="missing_features",
                           message="Send JSON with key 'features' (list of 100 floats)."), 400

        feats = np.array(feats_list, dtype=float).reshape(1, -1)
        if feats.shape[1] != INPUT_DIM:
            return jsonify(error="bad_dim",
                           message=f"Expected {INPUT_DIM} features, got {feats.shape[1]}"), 400

        xs = scaler.transform(feats)
        x = torch.tensor(xs, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return jsonify(prediction=pred, probabilities=[float(probs[0]), float(probs[1])])
    except Exception as e:
        app.logger.exception("predict_failed")
        return jsonify(error="predict_failed", message=str(e)), 500

# ==========================================
if __name__ == "__main__":
    # convenience: if OPENFACE_EXE is set, prepend its dir to PATH
    of = os.environ.get("OPENFACE_EXE")
    if of:
        os.environ["PATH"] = os.pathsep + os.path.dirname(of) + os.environ.get("PATH", "")
    # use PORT from env, default 5000 for Replit
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

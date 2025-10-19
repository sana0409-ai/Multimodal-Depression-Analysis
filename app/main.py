from flask import Flask, request, jsonify, render_template_string
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
    "hi i'm ellie thanks for coming in today",
    "i was created to talk to people in a safe and secure environment",
    "think of me as a friend i don't judge i can't i'm a computer",
    "i'm here to learn about people and would love to learn about you",
    "i'll ask a few questions to get us started and please feel free to tell me anything",
    "how are you doing today",
    "that's good",
    "where are you from originally",
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

def extract_clnf_features_from_video(video_path):
    """
    Run OpenFace FeatureExtraction with a timeout.
    Returns a 136-dim mean (x_*, y_*) vector or zeros on failure/timeout/skip.
    """
    if OPENFACE_SKIP or not shutil.which(OPENFACE_EXE):
        return np.zeros(136, dtype=np.float32)

    out_dir = tempfile.mkdtemp()
    mp4_to_delete = None
    try:
        src_for_of, mp4_to_delete = ffmpeg_transcode_to_mp4(video_path)
        subprocess.run(
            [OPENFACE_EXE, "-f", src_for_of, "-out_dir", out_dir, "-q"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=OPENFACE_TIMEOUT,
        )
        csvs = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
        if not csvs:
            return np.zeros(136, dtype=np.float32)
        df = pd.read_csv(os.path.join(out_dir, csvs[0]))
        landmark_cols = [c for c in df.columns if c.startswith("x_") or c.startswith("y_")]
        if not landmark_cols:
            return np.zeros(136, dtype=np.float32)
        return df[landmark_cols].mean(axis=0).astype(np.float32).values
    except subprocess.TimeoutExpired:
        app.logger.warning("OpenFace timed out — returning zeros for CLNF.")
        return np.zeros(136, dtype=np.float32)
    except Exception as e:
        app.logger.warning(f"OpenFace failed ({e}) — returning zeros for CLNF.")
        return np.zeros(136, dtype=np.float32)
    finally:
        try:
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f))
            os.rmdir(out_dir)
        except Exception:
            pass
        if mp4_to_delete and os.path.exists(mp4_to_delete):
            try:
                os.remove(mp4_to_delete)
            except Exception:
                pass

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

# ---------- UI ----------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Depression Model Live</title>
  <style>
    body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:20px}
    .q{font-size:18px;margin:10px 0}
    button{padding:10px 16px;border-radius:10px;border:1px solid #ddd;cursor:pointer}
    #log{white-space:pre-wrap;background:#fafafa;border:1px solid #eee;padding:10px;border-radius:8px;margin-top:12px;height:160px;overflow:auto}
    video{width:480px;border-radius:12px;border:1px solid #eee}
  </style>
</head>
<body>
  <h2>🎥🎤 Depression Screening (demo)</h2>
  <p>We’ll ask <b>{{nq}}</b> short questions. When you hit <b>Record</b>, we capture <b>{{secs}}s</b> of webcam + mic, upload, extract features, and move on.</p>

  <div class="q"><b>Question <span id="qnum">1</span>:</b> <span id="qtext"></span></div>

  <video id="preview" autoplay muted playsinline></video><br/><br/>
  <button id="btn">Record {{secs}}s</button>
  <button id="skip">Skip</button>

  <div id="log"></div>

  <script>
    const QUESTIONS = {{questions|tojson}};
    const SEC = {{secs}};
    const SID = crypto.randomUUID();
    let idx = 0, mediaStream=null, busy=false;

    const qnum = document.getElementById('qnum');
    const qtext = document.getElementById('qtext');
    const btn = document.getElementById('btn');
    const skip = document.getElementById('skip');
    const log = document.getElementById('log');
    const preview = document.getElementById('preview');

    function setQ() {
      qnum.textContent = (idx+1);
      qtext.textContent = QUESTIONS[idx];
    }
    setQ();

    async function getStream(){
      if (!mediaStream) {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video:{ width:{ideal:320}, height:{ideal:240} },
          audio:true
        });
        preview.srcObject = mediaStream;
      }
      return mediaStream;
    }

    function append(msg){ log.textContent += msg + "\\n"; log.scrollTop = log.scrollHeight; }

    async function recordOnce(){
      if (busy) return;
      busy = true; btn.disabled = true; skip.disabled = true;

      try {
        const stream = await getStream();
        const rec = new MediaRecorder(stream, {mimeType: 'video/webm;codecs=vp8,opus'});
        let chunks = [];
        rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
        rec.start();
        append("🎙️ Recording...");
        await new Promise(r => setTimeout(r, SEC*1000));
        rec.stop();
        await new Promise(r => rec.onstop = r);
        const blob = new Blob(chunks, {type: 'video/webm'});
        append("⬆️ Uploading " + Math.round(blob.size/1024) + " KB");
        const fd = new FormData();
        fd.append('sid', SID);
        fd.append('qidx', idx);
        fd.append('file', blob, 'seg.webm');

        // Abort if server stalls
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), 30000);

        let ok = false;
        try {
          const res = await fetch('/api/segment', {method:'POST', body:fd, signal: ctrl.signal});

          clearTimeout(t);
          const j = await res.json().catch(() => ({}));
          ok = res.ok && j && j.ok;
          if (ok) append("✅ Segment saved (" + j.count + "/" + QUESTIONS.length + ")");
          else append("❌ Segment failed — skipping this question.");
        } catch (err) {
          clearTimeout(t);
          append("❌ Upload/record error: " + err);
        }

        idx++;
        if (idx < QUESTIONS.length) { setQ(); }
        else { finalize(); }
      } finally {
        busy = false; btn.disabled = false; skip.disabled = false;
      }
    }

    async function finalize(){
      append("🧠 Finalizing prediction...");
      const res = await fetch('/api/finalize?sid=' + SID);

      const j = await res.json().catch(()=>({}));
      if (!res.ok || !j) { append("❌ Finalize failed."); return; }
      append("🎯 P(Depressed)=" + j.p_depressed.toFixed(3) + " → " + j.label);
    }

    btn.onclick = recordOnce;
    skip.onclick = () => { if (!busy) { idx++; if (idx<QUESTIONS.length) setQ(); else finalize(); } };
  </script>
</body>
</html>
"""

# ----- HF Embeddings (offload to Inference API) -----
HF_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", None)
EMB_DIM = 384  # MiniLM-L6-v2

def get_text_embedding(text: str) -> np.ndarray:
    """
    Calls HF Inference API for embeddings. Returns a 384-dim vector.
    Falls back to zeros on error or missing token so the app keeps running.
    """
    if not text or not text.strip():
        return np.zeros(EMB_DIM, dtype=np.float32)
    if not HF_API_TOKEN:
        return np.zeros(EMB_DIM, dtype=np.float32)

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        resp = requests.post(url, headers=headers, json={"inputs": text, "truncate": True}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # mean pool if token-level output
        if isinstance(data, list) and data and isinstance(data[0], list):
            vec = np.array(data, dtype=np.float32).mean(axis=0)
        else:
            vec = np.array(data, dtype=np.float32)
        if vec.shape[0] < EMB_DIM:
            vec = np.concatenate([vec, np.zeros(EMB_DIM - vec.shape[0], dtype=np.float32)])
        elif vec.shape[0] > EMB_DIM:
            vec = vec[:EMB_DIM]
        return vec.astype(np.float32)
    except Exception as e:
        app.logger.warning(f"HF embedding failed: {e}")
        return np.zeros(EMB_DIM, dtype=np.float32)

# ---------- routes ----------
@app.get("/")
def ui():
    return render_template_string(INDEX_HTML, questions=QUESTIONS, secs=SEG_SECONDS, nq=len(QUESTIONS))

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
        openface_present=bool(of_path),
        openface_exe=of_path or OPENFACE_EXE,
        openface_timeout=OPENFACE_TIMEOUT,
        openface_skip=OPENFACE_SKIP,
        stt_skip=STT_SKIP,
        hf_embeddings=bool(HF_API_TOKEN),
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

        xs = scaler.transform(x)
        t  = torch.from_numpy(xs).float()
        with torch.no_grad():
            out = model(t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            p_dep = float(probs[1])
            label = "Depressed" if p_dep > 0.5 else "Not Depressed"

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
    # use PORT from env for Render/Railway, default 7860 locally
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)

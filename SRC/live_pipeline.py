import os
import cv2
import joblib
import torch
import tempfile
import subprocess
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import opensmile
import parselmouth
import time
from pathlib import Path
from torch import nn
from sentence_transformers import SentenceTransformer

OPENFACE_EXE = "FeatureExtraction.exe"  

SMILE_GEMAPS = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

def build_model(input_dim, hidden=32, drop=0.2):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.LeakyReLU(0.1),
        nn.Dropout(drop),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.LeakyReLU(0.1),
        nn.Dropout(drop),
        nn.Linear(hidden, 2),
    )

def buffer_to_wav(audio_buf: np.ndarray, sr: int) -> str:
    tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tf.name, audio_buf, sr)
    return tf.name

def extract_gemaps(audio_buffer, sr):
    df = SMILE_GEMAPS.process_signal(audio_buffer, sr)
    return df.mean(axis=0).astype(np.float32).values  # shape ~ (88,)

def extract_formants(audio_buffer, sr):
    sound = parselmouth.Sound(audio_buffer, sampling_frequency=sr)
    formant = sound.to_formant_burg()
    duration = sound.get_total_duration()
    times = np.linspace(0, duration, num=100)
    formant_features = []
    for t in times:
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        formant_features.append([f1, f2, f3])
    return np.nanmean(formant_features, axis=0)  # shape (3,)

def extract_clnf_features_from_video(video_path):
    out_dir = tempfile.mkdtemp()
    try:
        subprocess.run([OPENFACE_EXE, "-f", video_path, "-out_dir", out_dir], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        csvs = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
        if not csvs:
            print(f"No CSV output from OpenFace for {video_path}")
            return np.zeros(136, dtype=np.float32)
        fname = os.path.join(out_dir, csvs[0])
        df = pd.read_csv(fname)
        landmark_cols = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
        if not landmark_cols:
            print("No landmark columns found in OpenFace output.")
            return np.zeros(136, dtype=np.float32)
        df = df[landmark_cols]
        return df.mean(axis=0).astype(np.float32).values
    except Exception as e:
        print(f"Error extracting CLNF features: {e}")
        return np.zeros(136, dtype=np.float32)
    finally:
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir, f))
        os.rmdir(out_dir)

def transcribe_wav(wav_path: str, recognizer: sr.Recognizer) -> str:
    with sr.AudioFile(wav_path) as src:
        audio = recognizer.record(src)
    try:
        return recognizer.recognize_google(audio)
    except (sr.RequestError, sr.UnknownValueError):
        return ""

def record_av_segment(video_file, audio_file, duration_sec=5, cam_index=0, sr=16000):
    cap = cv2.VideoCapture(cam_index)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_w = int(cap.get(3))
    frame_h = int(cap.get(4))
    out = cv2.VideoWriter(video_file, fourcc, 20, (frame_w, frame_h))
    audio_frames = []

    def audio_callback(indata, frames, t0, status):
        audio_frames.append(indata.copy())

    stream = sd.InputStream(samplerate=sr, channels=1, callback=audio_callback)
    stream.start()
    start = time.time()
    print("Recording (look at the camera and answer)...")
    while (time.time() - start) < duration_sec:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    stream.stop()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    audio_np = np.concatenate(audio_frames, axis=0).flatten()
    sf.write(audio_file, audio_np, sr)
    return audio_np

def extract_live_q_vectors(video_segs, audio_segs, text_segs, bert_model, sr):
    q_vecs = []
    for video_file, audio_buf, text in zip(video_segs, audio_segs, text_segs):
        c = extract_gemaps(audio_buf, sr)
        f = extract_formants(audio_buf, sr)
        cl = extract_clnf_features_from_video(video_file)
        emb = bert_model.encode([text], convert_to_numpy=True)[0]
        print("GeMAPS shape:", c.shape, "Formant shape:", f.shape, "CLNF shape:", cl.shape, "BERT shape:", emb.shape)
        q_vecs.append(np.concatenate([c, f, cl, emb], axis=0))
    return np.stack(q_vecs, axis=0)

def generate_feature_names(D: int, num_q: int):
    return [f"q{q+1}_f{f}" for q in range(num_q) for f in range(D)]

def main():
    base = Path(__file__).resolve().parent.parent

    questions = [
        "hi i'm ellie thanks for coming in today",
        "i was created to talk to people in a safe and secure environment",
        "think of me as a friend i don't judge i can't i'm a computer",
        "i'm here to learn about people and would love to learn about you",
        "i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential",
        "how are you doing today",
        "that's good",
        "where are you from originally"
    ]

    top100     = joblib.load(base/'models'/'top100_features.joblib')
    scaler     = joblib.load(base/'models'/'scaler.joblib')
    bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    model      = build_model(len(top100))
    state      = torch.load(base/'models'/'depression_model.pt', map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    recognizer = sr.Recognizer()
    sr_rate = 16000
    cam_index = 0
    segment_duration = 5  # seconds

    video_segs, audio_segs, text_segs = [], [], []
    print("Answer each question in turn. When ready, press ENTER to record (5 seconds)...\n")

    for idx, q in enumerate(questions):
        print(f"\nQ{idx+1}: {q}")
        ans = input("Press ENTER when ready to record, or type 'q' to quit: ")
        if ans.strip().lower() == 'q':
            print("Exiting early.")
            return
        tmp_video = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
        tmp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        video_path = tmp_video.name
        audio_path = tmp_audio.name
        tmp_video.close()
        tmp_audio.close()
        audio_buf = record_av_segment(video_path, audio_path, duration_sec=segment_duration,
                                      cam_index=cam_index, sr=sr_rate)
        txt = transcribe_wav(audio_path, recognizer)
        print(f" Auto-transcript: {txt}")
        video_segs.append(video_path)
        audio_segs.append(audio_buf)
        text_segs.append(txt)
        os.remove(audio_path)
        print(f"Captured {idx+1}/{len(questions)}")

    q_vecs = extract_live_q_vectors(video_segs, audio_segs, text_segs, bert_model, sr_rate)
    # Remove temp video files
    for v in video_segs:
        if os.path.exists(v):
            os.remove(v)

    flat   = q_vecs.reshape(-1)
    D      = q_vecs.shape[1]
    names  = generate_feature_names(D, len(questions))
    name2val = dict(zip(names, flat))

    top100 = list(top100)
    missing = [f for f in top100 if f not in name2val]
    if missing:
        print("Missing features:", missing)

    x  = np.array([name2val[f] for f in top100], dtype=np.float32).reshape(1, -1)
    xs = scaler.transform(x)
    t  = torch.from_numpy(xs).float()
    with torch.no_grad():
        out = model(t)
        p   = torch.softmax(out, 1)[0,1].item()
        lab = "Depressed" if p > 0.5 else "Not Depressed"

    print(f"\n Final → P(depressed)={p:.2f}, {lab}")

if __name__ == "__main__":
    main()

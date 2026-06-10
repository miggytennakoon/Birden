from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, subprocess, collections, builtins
from typing import Optional

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── LIVE LOG BUFFER ──
_log_buffer = collections.deque(maxlen=100)
_orig_print = builtins.print

def _print(*args, **kwargs):
    _log_buffer.append(" ".join(str(a) for a in args))
    _orig_print(*args, **kwargs)

builtins.print = _print

# ── PATCH birdnetlib to use tf.lite instead of tflite_runtime ──
# This runs before birdnetlib is imported, replacing the broken
# tflite_runtime with tensorflow's built-in lite interpreter.
try:
    import tensorflow as tf
    import sys
    # Create a fake tflite_runtime module pointing to tf.lite
    import types
    tflite_runtime = types.ModuleType("tflite_runtime")
    tflite_runtime.interpreter = tf.lite
    interpreter_mod = types.ModuleType("tflite_runtime.interpreter")
    interpreter_mod.Interpreter = tf.lite.Interpreter
    interpreter_mod.load_delegate = None
    sys.modules["tflite_runtime"] = tflite_runtime
    sys.modules["tflite_runtime.interpreter"] = interpreter_mod
    print("✅ TFLite patched via tensorflow.lite")
except Exception as e:
    print(f"⚠️ TFLite patch failed: {e}")

# ── PRE-LOAD ANALYZER ONCE at startup ──
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from birdnetlib.analyzer import Analyzer
        _analyzer = Analyzer()
        print("✅ BirdNET analyzer loaded successfully")
        return _analyzer
    except Exception as e:
        print(f"❌ Failed to load analyzer: {e}")
        return None

@app.on_event("startup")
def startup_event():
    get_analyzer()

NOISE_LABELS = {
    "fireworks", "engine", "power tools", "human non-vocal", "human vocal",
    "human whistle", "noise", "silence", "aircraft", "car", "siren",
    "dog", "cat", "frog", "insect", "rain", "wind", "water", "thunder",
    "music", "gun", "chainsaw", "bell", "crowd"
}

@app.get("/")
def health():
    return {"status": "ok", "analyzer_ready": _analyzer is not None}

@app.get("/logs")
def get_logs():
    return {"logs": list(_log_buffer)}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0)
):
    data = await file.read()
    print(f"📥 Received {len(data)} bytes, lat={lat}, lon={lon}")

    a = get_analyzer()
    if a is None:
        print("❌ Analyzer not available")
        return {"detections": [], "error": "Analyzer failed to load"}

    suffix = ".webm"
    if file.filename:
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext in (".mp4", ".ogg", ".wav", ".m4a", ".aac"):
            suffix = ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    wav_path = tmp_path.replace(suffix, ".wav")

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-af", "highpass=f=100,lowpass=f=15000",
             "-ar", "48000", "-ac", "1", wav_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"❌ ffmpeg error: {result.stderr[-300:]}")
            return {"detections": []}

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            print(f"❌ WAV missing or too small")
            return {"detections": []}

        print(f"🎵 WAV ready: {os.path.getsize(wav_path)} bytes")

        from birdnetlib import Recording

        if lat != 0.0 or lon != 0.0:
            r = Recording(a, wav_path, lat=lat, lon=lon, min_conf=0.07)
        else:
            r = Recording(a, wav_path, min_conf=0.07)

        r.analyze()

        bird_detections = sorted(
            [d for d in r.detections if d.get("common_name", "").lower() not in NOISE_LABELS],
            key=lambda d: d.get("confidence", 0),
            reverse=True
        )[:10]

        if bird_detections:
            top = bird_detections[0]
            print(f"🐦 Top match: {top.get('common_name')} ({round(top.get('confidence', 0) * 100)}% confidence)")
            print(f"📋 All: {[(d.get('common_name'), str(round(d.get('confidence',0)*100))+'%') for d in bird_detections]}")
        else:
            print(f"🔇 No birds detected in this clip")

        return {"detections": bird_detections}

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return {"detections": [], "error": str(e)}

    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        if os.path.exists(wav_path): os.unlink(wav_path)

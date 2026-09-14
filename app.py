from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("CNN_MODEL_PATH", BASE_DIR / "model_cnn_brokoli_kembangkol.keras"))
IMG_SIZE = (150, 150)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Model dimuat sekali saat aplikasi Python start, bukan setiap request.
model = tf.keras.models.load_model(MODEL_PATH)


def json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "message": message}), status


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "BroKol CNN API",
        "model": MODEL_PATH.name,
        "input": "150x150 RGB",
        "classes": ["BROKOLI", "KEMBANG KOL"],
    })


@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "ready", "model_loaded": True})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return json_error("Tidak ada gambar yang dikirim.")

    uploaded = request.files["image"]
    if not uploaded or not uploaded.filename:
        return json_error("Silakan pilih gambar terlebih dahulu.")

    if uploaded.mimetype not in ALLOWED_MIME:
        return json_error("Format gambar harus JPG, PNG, atau WEBP.", 415)

    raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        return json_error("File gambar kosong.")
    if len(raw) > MAX_UPLOAD_BYTES:
        return json_error("Ukuran gambar maksimal 8 MB.", 413)

    try:
        # Sama dengan notebook: RGB -> resize 150x150 -> float array -> batch dimension.
        # Rescaling 1/255 SUDAH berada di dalam model, jadi tidak dinormalisasi ulang di sini.
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize(IMG_SIZE, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32)
        image_input = np.expand_dims(image_array, axis=0)

        score = float(model.predict(image_input, verbose=0)[0][0])

        # Mapping asli notebook:
        # 0 = brokoli, 1 = kembangkol, threshold = 0.5
        if score >= 0.5:
            label = "KEMBANG KOL"
            class_index = 1
            confidence = score * 100.0
        else:
            label = "BROKOLI"
            class_index = 0
            confidence = (1.0 - score) * 100.0

        return jsonify({
            "ok": True,
            "label": label,
            "class_index": class_index,
            "confidence": round(confidence, 2),
            "raw_score": round(score, 6),
            "threshold": 0.5,
            "input_size": [150, 150, 3],
        })

    except UnidentifiedImageError:
        return json_error("File tidak dapat dibaca sebagai gambar.", 415)
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return json_error(f"Prediksi gagal diproses: {type(exc).__name__}", 500)


@app.errorhandler(413)
def file_too_large(_):
    return json_error("Ukuran gambar maksimal 8 MB.", 413)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)

"""Tahap 6: export model 1 ke TFLite (float16) + verifikasi numerik.

Output: output/05_bahan_aplikasi/model.tflite, labels.json, model_meta.json
"""
import json
import shutil
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OUT, save_json, tampil  # noqa: E402

SRC = OUT / "01_cnn"
D = OUT / "05_bahan_aplikasi"
D.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(SRC / "model.keras")
IMG = int(model.input_shape[1])
labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
print(f"model dimuat | input {model.input_shape} | kelas {len(labels)}", flush=True)

# ---------- konversi ----------
import contextlib
import io

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
_log = io.StringIO()
with contextlib.redirect_stdout(_log), contextlib.redirect_stderr(_log):
    tflite_bytes = conv.convert()
(D / "model.tflite").write_bytes(tflite_bytes)
size_mb = len(tflite_bytes) / 1024 ** 2
print(f"TFLite float16: {size_mb:.2f} MB", flush=True)

# ---------- verifikasi ----------
interp = tf.lite.Interpreter(model_path=str(D / "model.tflite"))
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
print(f"input {inp_det['shape']} {inp_det['dtype']} | output {out_det['shape']}", flush=True)

te = pd.read_csv(DATA / "test.csv")
rng = np.random.default_rng(42)
sample = te.iloc[rng.choice(len(te), size=200, replace=False)]


def load_raw(p):
    raw = tf.io.read_file(str(p))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG, IMG])
    return tf.cast(img, tf.float32).numpy()[None, ...]


keras_probs, lite_probs = [], []
for p in sample.path.values:
    x = load_raw(p)
    kp = model.predict(x, verbose=0)[0]
    interp.set_tensor(inp_det["index"], x)
    interp.invoke()
    lp = interp.get_tensor(out_det["index"])[0]
    keras_probs.append(kp)
    lite_probs.append(lp)

kp = np.array(keras_probs)
lp = np.array(lite_probs)
agree = float((kp.argmax(1) == lp.argmax(1)).mean())
maxdiff = float(np.abs(kp - lp).max())
meandiff = float(np.abs(kp - lp).mean())
print(f"kesepakatan argmax: {agree:.3%} | max diff {maxdiff:.5f} | mean diff {meandiff:.6f}", flush=True)

y = sample.y.values
acc_keras = float((kp.argmax(1) == y).mean())
acc_lite = float((lp.argmax(1) == y).mean())
print(f"akurasi subset 200 gambar: keras {acc_keras:.3%} | tflite {acc_lite:.3%}", flush=True)

# ---------- aset untuk teman ----------
shutil.copy(DATA / "labels.json", D / "labels.json")
shutil.copy(SRC / "metrics.json", D / "metrics_model_1.json")

meta = {
    "input": {"shape": [1, IMG, IMG, 3], "dtype": "float32",
              "range": "0..255 (jangan dinormalisasi lagi, preprocessing ada di dalam model)",
              "color": "RGB"},
    "output": {"shape": [1, len(labels)], "dtype": "float32", "activation": "softmax",
               "meaning": "probabilitas per kelas, urutan sesuai labels.json"},
    "quantization": "float16 weights",
    "size_mb": round(size_mb, 2),
    "verification": {"n_images": len(sample), "argmax_agreement": agree,
                     "max_abs_diff": maxdiff, "mean_abs_diff": meandiff,
                     "acc_keras_subset": acc_keras, "acc_tflite_subset": acc_lite},
    "labels_count": len(labels),
}
save_json(D / "model_meta.json", meta)

tampil("Verifikasi Keras vs TFLite", pd.DataFrame({
    "Pemeriksaan": ["Kesepakatan argmax (200 gambar)", "Selisih probabilitas maksimum",
                    "Akurasi subset - Keras", "Akurasi subset - TFLite", "Ukuran model"],
    "Hasil": [f"{agree:.2%}", f"{maxdiff:.5f}",
              f"{acc_keras:.1%}", f"{acc_lite:.1%}", f"{size_mb:.2f} MB"],
}))

# --- tabel kontrak input/output ---
tampil("Kontrak input/output", pd.DataFrame({
    "Bagian": ["Input", "Input", "Input", "Output"],
    "Ketentuan": [f"bentuk [1, {IMG}, {IMG}, 3]", "float32, nilai 0-255 (jangan dibagi 255)",
                  "RGB, resize bilinear", f"[1, {len(labels)}] softmax, urutan = labels.json"],
}))
print("DONE TAHAP 6", flush=True)

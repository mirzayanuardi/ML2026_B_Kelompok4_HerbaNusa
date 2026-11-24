"""Tahap 6: export model 1 ke TFLite (float16) + verifikasi numerik.

Output: output/05_bahan_aplikasi/model.tflite, labels.json, README.md
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OUT, save_json, write_md  # noqa: E402

SRC = OUT / "01_cnn"
D = OUT / "05_bahan_aplikasi"
D.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(SRC / "model.keras")
IMG = int(model.input_shape[1])
labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
print(f"model dimuat | input {model.input_shape} | kelas {len(labels)}", flush=True)

# ---------- konversi ----------
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
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

sections = [
    ("Isi folder", f"""| File | Isi |
|---|---|
| `model.tflite` | Model siap pakai, input float32 0..255 ukuran 1x{IMG}x{IMG}x3 |
| `labels.json` | 100 nama kelas, urutan sama dengan output model |
| `model_meta.json` | Spesifikasi input/output + hasil verifikasi |
| `metrics_model_1.json` | Angka akurasi model |
| `safety_rules.json` | Nilai AMBANG + aturan penolakan |"""),
    ("Cara pakai di Flutter", f"""Tambahkan `tflite_flutter` di `pubspec.yaml`, taruh `model.tflite` dan
`labels.json` di folder `assets/`.

Alur inferensi:

1. Ambil gambar (kamera/galeri), decode jadi RGB.
2. Resize ke **{IMG}x{IMG}** (bilinear).
3. Konversi ke float32 dengan nilai **0..255** — JANGAN dibagi 255.
   Normalisasi sudah ada di dalam model (`Rescaling(1/127.5, offset=-1)`).
4. Bentuk tensor `[1, {IMG}, {IMG}, 3]`, jalankan interpreter.
5. Output `[1, 100]` = probabilitas, urutannya sesuai `labels.json`.
6. Ambil 3 nilai terbesar -> top-3 kandidat.
7. Tolak menjawab kalau probabilitas tertinggi < `AMBANG` (lihat `safety_rules.json`)."""),
    ("Verifikasi", f"""| Pemeriksaan | Hasil |
|---|---|
| Kesepakatan argmax Keras vs TFLite (200 gambar) | {agree:.2%} |
| Selisih probabilitas maksimum | {maxdiff:.5f} |
| Ukuran model | {size_mb:.2f} MB |

Kalau teman Anda mendapat hasil berbeda jauh di HP, kemungkinan besar
preprocessing-nya tidak sama (resize atau rentang nilai)."""),
    ("Catatan penting", """- Model ini **hanya** mengenali 100 spesies di `labels.json`.
- Tanaman di luar daftar tetap akan dipaksa masuk ke salah satu kelas,
  karena itu aturan penolakan berbasis confidence WAJIB dipakai.
- Jangan menampilkan satu jawaban saja; selalu tampilkan top-3."""),
]
write_md(D / "README_MODEL.md", "Model TFLite untuk Aplikasi", sections)
print("DONE TAHAP 6", flush=True)

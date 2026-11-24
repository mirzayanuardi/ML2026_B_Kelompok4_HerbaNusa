"""Evaluasi TTA (test-time augmentation) untuk model yang sudah dilatih.

TTA = menjalankan model beberapa kali pada variasi gambar yang sama,
lalu merata-ratakan probabilitasnya. Tidak perlu training ulang.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, OUT, load_labels, load_tags, safety_metrics, save_json, topk_acc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="01_cnn")
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

S = OUT / args.src
D = OUT / (args.src + "_tta")
D.mkdir(parents=True, exist_ok=True)
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(S / "model.keras")
IMG = int(model.input_shape[1])
labels = load_labels()
te = pd.read_csv(DATA / "test.csv")
yte = te.y.values
print(f"model {S.name} | img {IMG} | test {len(te)}", flush=True)


def decode_raw(path):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG, IMG])
    return tf.cast(tf.ensure_shape(img, [IMG, IMG, 3]), tf.float32)


def view_original(x):
    return x


def view_flip(x):
    return tf.image.flip_left_right(x)


def view_zoom(a):
    def f(x):
        h = int(IMG * a)
        off = (IMG - h) // 2
        y = tf.image.crop_to_bounding_box(x, off, off, h, h)
        return tf.image.resize(y, [IMG, IMG])
    return lambda x: f(tf.convert_to_tensor(x))


base = tf.data.Dataset.from_tensor_slices(te.path.values)
base = base.map(decode_raw, num_parallel_calls=tf.data.AUTOTUNE)

views = {"asli": lambda x: x,
         "flip": view_flip,
         "zoom90": view_zoom(0.90),
         "zoom80": view_zoom(0.80),
         "flip_zoom90": lambda x: view_zoom(0.90)(view_flip(x))}

all_probs = []
for name, fn in views.items():
    ds = base.map(fn, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch).prefetch(tf.data.AUTOTUNE)
    p = model.predict(ds, verbose=0)
    acc = float((p.argmax(1) == yte).mean())
    print(f"  view {name:12} top-1 {acc:.4f}", flush=True)
    all_probs.append(p)

avg = np.mean(all_probs, axis=0)
np.save(D / "probabilitas_test_tta.npy", avg)
pred = avg.argmax(1)
top1 = float((pred == yte).mean())
top3 = topk_acc(yte, avg, 3)
print(f"\nTTA gabungan: top-1 {top1:.4f} | top-3 {top3:.4f}", flush=True)

tags = load_tags()
safe, _ = safety_metrics(yte, pred, labels, tags)
save_json(D / "metrics_tta.json", {"top1": top1, "top3": top3,
                                   "views": list(views), **safe})

# figur perbandingan
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

single = np.load(S / "probabilitas_test.npy")
fig, ax = plt.subplots(figsize=(6, 4))
vals = [float((single.argmax(1) == yte).mean()), top1]
ax.bar(["tanpa TTA", "dengan TTA (5 view)"], [v * 100 for v in vals], color=["#90a4ae", "#1565c0"])
for i, v in enumerate(vals):
    ax.text(i, v * 100 + 0.3, f"{v*100:.2f}%", ha="center")
ax.set_ylim(70, 90); ax.set_ylabel("Top-1 (%)")
ax.set_title("Efek TTA (tanpa training ulang)"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "tta_vs_single.png", dpi=130); plt.close()
print("DONE TTA", flush=True)

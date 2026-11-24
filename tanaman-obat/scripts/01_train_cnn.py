"""Tahap 2: Model 1 - CNN MobileNetV2 fine-tune (100 kelas).

Simpan ke output/01_cnn/:
  model.keras, history.csv, probabilitas_test.npy, pred_test.npy,
  emb_train.npy, emb_val.npy, emb_test.npy, y_*.npy,
  figures/*.png, metrics.json, HASIL.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from common import (DATA, OUT, abstain_curve, load_tags, safety_metrics,  # noqa: E402
                    save_json, topk_acc, worst_classes, write_md)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

tf.random.set_seed(42)
np.random.seed(42)

ap = argparse.ArgumentParser()
ap.add_argument("--img", type=int, default=128)
ap.add_argument("--head", type=int, default=3)
ap.add_argument("--fine", type=int, default=12)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out", default="01_cnn")
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--ls", type=float, default=0.0, help="label smoothing")
ap.add_argument("--lr-fine", type=float, default=3e-4)
ap.add_argument("--reuse", action="store_true")
args = ap.parse_args()

D = OUT / args.out
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)
IMG = args.img
BATCH = args.batch

labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
N = len(labels)
tr = pd.read_csv(DATA / "train.csv")
va = pd.read_csv(DATA / "val.csv")
te = pd.read_csv(DATA / "test.csv")
print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | kelas {N} | img {IMG}", flush=True)


def decode(path, y):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG, IMG])
    img = tf.ensure_shape(img, [IMG, IMG, 3])
    return tf.cast(img, tf.float32), y


aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.12),
    tf.keras.layers.RandomZoom(0.12, 0.10),
    tf.keras.layers.RandomContrast(0.15),
    tf.keras.layers.RandomBrightness(0.12),
], name="aug")


def make_ds(frame, training, shuffle=2048):
    ds = tf.data.Dataset.from_tensor_slices((frame.path.values, frame.y.values))
    if training:
        ds = ds.shuffle(shuffle, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    if args.ls > 0:
        ds = ds.map(lambda x, y: (x, tf.one_hot(y, N)),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build():
    inp = tf.keras.Input((IMG, IMG, 3))
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inp)
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(x)
    feats = x
    x = tf.keras.layers.Dropout(args.dropout)(x)
    out = tf.keras.layers.Dense(N, activation="softmax", name="logits")(x)
    return tf.keras.Model(inp, out), tf.keras.Model(inp, feats), base


ds_tr = make_ds(tr, True)
ds_va = make_ds(va, False)
ds_te = make_ds(te, False)

model_path = D / "model.keras"
if args.reuse and model_path.exists():
    print("reuse model lama", flush=True)
    model = tf.keras.models.load_model(model_path)
    base = [l for l in model.layers if l.name.startswith("mobilenet")][0]
    feat_model = tf.keras.Model(model.input, model.get_layer("embedding").output)
    hist = pd.read_csv(D / "history.csv")
else:
    model, feat_model, base = build()
    LOSS = (tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.ls)
            if args.ls > 0 else "sparse_categorical_crossentropy")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=LOSS, metrics=["accuracy"])
    print(f"parameter total: {model.count_params():,}", flush=True)

    t0 = time.time()
    h1 = model.fit(ds_tr, validation_data=ds_va, epochs=args.head, verbose=2)
    print(f"fase head: {(time.time()-t0)/60:.1f} menit", flush=True)

    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_fine),
                  loss=LOSS, metrics=["accuracy"])
    print(f"parameter trainable: {sum(np.prod(w.shape) for w in model.trainable_weights):,}", flush=True)

    t0 = time.time()
    h2 = model.fit(ds_tr, validation_data=ds_va, epochs=args.fine, verbose=2,
                   callbacks=[
                       tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                                           patience=2, min_lr=1e-6),
                   ])
    print(f"fase fine-tune: {(time.time()-t0)/60:.1f} menit", flush=True)
    hist = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
    hist.to_csv(D / "history.csv", index=False)
    model.save(model_path)

# ---------- evaluasi ----------
print("prediksi test...", flush=True)
probs = model.predict(ds_te, verbose=0)
probs_va = model.predict(ds_va, verbose=0)
emb_tr = feat_model.predict(make_ds(tr, False), batch_size=64, verbose=0)
emb_va = feat_model.predict(ds_va, batch_size=64, verbose=0)
emb_te = feat_model.predict(ds_te, batch_size=64, verbose=0)

y_te, y_va, y_tr = te.y.values, va.y.values, tr.y.values
pred = probs.argmax(1)

np.save(D / "probabilitas_test.npy", probs)
np.save(D / "probabilitas_val.npy", probs_va)
np.save(D / "pred_test.npy", pred)
np.save(D / "emb_train.npy", emb_tr)
np.save(D / "emb_val.npy", emb_va)
np.save(D / "emb_test.npy", emb_te)
np.save(D / "y_train.npy", y_tr)
np.save(D / "y_val.npy", y_va)
np.save(D / "y_test.npy", y_te)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score  # noqa: E402

top1 = accuracy_score(y_te, pred)
top3 = topk_acc(y_te, probs, 3)
macro = f1_score(y_te, pred, average="macro")
rec_bucket = recall_score(y_te, pred, average=None, labels=list(range(N)))
tags = load_tags()
safe, fatal = safety_metrics(y_te, pred, labels, tags)

metrics = {"model": "CNN MobileNetV2", "img": IMG, "n_params": int(model.count_params()),
           "top1": float(top1), "top3": float(top3), "macro_f1": float(macro),
           **safe}
save_json(D / "metrics.json", metrics)

print(f"\nTOPI-1 {top1:.4f} | TOP-3 {top3:.4f} | macroF1 {macro:.4f}", flush=True)
print(f"fatal {safe['fatal_n']}/{safe['n_test_beracun']} = {safe['fatal_rate']:.1%}", flush=True)

# ---------- figur ----------
if "accuracy" in hist.columns:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["accuracy"], label="train", marker="o", ms=3)
    ax[0].plot(hist["val_accuracy"], label="val", marker="o", ms=3)
    ax[0].set_title("Akurasi per epoch"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(hist["loss"], label="train", marker="o", ms=3)
    ax[1].plot(hist["val_loss"], label="val", marker="o", ms=3)
    ax[1].set_title("Loss per epoch"); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=.3); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "kurva_training.png", dpi=130); plt.close()

cm = confusion_matrix(y_te, pred, labels=list(range(N)))
fig, ax = plt.subplots(figsize=(12, 11))
ax.imshow(cm, cmap="viridis")
ax.set_title(f"Confusion matrix CNN ({N} kelas)"); ax.set_xlabel("prediksi"); ax.set_ylabel("aktual")
plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png", dpi=110); plt.close()

per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
order = np.argsort(per_class)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(per_class)), per_class[order], color="#2e7d32")
ax.set_title("Recall per kelas (urut naik)"); ax.set_xlabel("kelas"); ax.set_ylabel("recall")
ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "recall_per_kelas.png", dpi=120); plt.close()

cur = abstain_curve(y_te, pred, probs.max(1))
cur.to_csv(D / "kurva_abstain.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(cur.coverage * 100, cur.accuracy * 100, marker="o", color="#1565c0")
ax.set_xlabel("Foto yang dijawab (%)"); ax.set_ylabel("Akurasi pada yang dijawab (%)")
ax.set_title("Kalau model ragu, jangan dijawab"); ax.invert_xaxis(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "kurva_abstain.png", dpi=130); plt.close()

wk, best = worst_classes(y_te, pred, labels, 15)
wk.to_csv(D / "kelas_terburuk.csv", index=False)

# ---------- HASIL.md ----------
tbl_abs = "\n".join(f"| {r.coverage:.0%} | {r.n} | {r.accuracy:.1%} |" for r in cur.itertuples())
tbl_worst = "\n".join(f"| {r.spesies} | {r.recall:.2f} | {r.n} |" for r in wk.itertuples())
tbl_best = "\n".join(f"| {r.spesies} | {r.recall:.2f} |" for r in best.itertuples())
sections = [
    ("Ringkasan", f"""| Metrik | Nilai |
|---|---|
| Top-1 | **{top1:.1%}** |
| Top-3 | **{top3:.1%}** |
| Macro-F1 | {macro:.1%} |
| Resolusi input | {IMG}x{IMG} |
| Parameter | {model.count_params():,} |
| Ukuran model | {model_path.stat().st_size/1024**2:.1f} MB |"""),
    ("Keamanan", f"""| Metrik | Nilai |
|---|---|
| Foto beracun di test | {safe['n_test_beracun']} |
| Dikenali sebagai beracun | {safe['recall_beracun_sebagai_beracun']:.1%} |
| **Kesalahan fatal (beracun -> spesies pangan)** | **{safe['fatal_n']} = {safe['fatal_rate']:.1%}** |
| Recall pangan | {safe['recall_pangan']:.1%} |
| Alarm beracun palsu pada pangan | {safe['alarm_palsu_beracun_pada_pangan']:.1%} |"""),
    ("Kalau ragu jangan dijawab", f"""| Cakupan | n | Akurasi |
|---|---|---|
{tbl_abs}"""),
    ("15 kelas terburuk", f"""| Spesies | Recall | n |
|---|---|---|
{tbl_worst}"""),
    ("10 kelas terbaik", f"""| Spesies | Recall |
|---|---|
{tbl_best}"""),
    ("Cara membaca", """- **Top-1**: tebakan peringkat 1 benar.
- **Top-3**: jawaban benar ada di 3 kandidat teratas (jaring pengaman aplikasi).
- **Kesalahan fatal**: foto tanaman beracun ditebak sebagai spesies yang aman dimakan.
  Ini metrik paling penting untuk aplikasi keselamatan lapangan."""),
]
write_md(D / "HASIL.md", "Model 1 - CNN MobileNetV2", sections)
print("DONE MODEL 1", flush=True)

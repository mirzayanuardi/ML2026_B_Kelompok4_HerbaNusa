"""Tahap 4: Model 3 - Random Forest 300 pohon di atas embedding CNN."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (OUT, abstain_curve, load_labels, load_tags, safety_metrics,  # noqa: E402
                    save_json, topk_acc, worst_classes, write_md)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = OUT / "01_cnn"
D = OUT / "03_rf"
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)

labels = load_labels()
N = len(labels)
Xtr = np.load(SRC / "emb_train.npy")
Xte = np.load(SRC / "emb_test.npy")
ytr = np.load(SRC / "y_train.npy")
yte = np.load(SRC / "y_test.npy")
print(f"embedding train {Xtr.shape} test {Xte.shape} | kelas {N}", flush=True)

t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1,
                            random_state=42, class_weight="balanced_subsample")
rf.fit(Xtr, ytr)
train_sec = time.time() - t0
print(f"fit RF: {train_sec/60:.1f} menit", flush=True)

probs = rf.predict_proba(Xte)
pred = probs.argmax(1)
top1 = accuracy_score(yte, pred)
top3 = topk_acc(yte, probs, 3)
macro = f1_score(yte, pred, average="macro")

np.save(D / "probabilitas_test.npy", probs)
np.save(D / "pred_test.npy", pred)

tags = load_tags()
safe, fatal = safety_metrics(yte, pred, labels, tags)
metrics = {"model": "Random Forest 300", "top1": float(top1), "top3": float(top3),
           "macro_f1": float(macro), "train_sec": round(train_sec, 1), **safe}
save_json(D / "metrics.json", metrics)
print(f"TOPI-1 {top1:.4f} | TOP-3 {top3:.4f} | macroF1 {macro:.4f}", flush=True)

cm = confusion_matrix(yte, pred, labels=list(range(N)))
fig, ax = plt.subplots(figsize=(12, 11))
ax.imshow(cm, cmap="cividis")
ax.set_title("Confusion matrix Random Forest (100 kelas)")
ax.set_xlabel("prediksi"); ax.set_ylabel("aktual")
plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.png", dpi=110); plt.close()

per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(per_class)), per_class[np.argsort(per_class)], color="#6a1b9a")
ax.set_title("Recall per kelas (urut naik) - Random Forest"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "recall_per_kelas.png", dpi=120); plt.close()

imp = pd.Series(rf.feature_importances_).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(30), imp.values[:30], color="#6a1b9a")
ax.set_title("30 fitur embedding paling penting"); ax.set_xlabel("peringkat fitur")
plt.tight_layout(); plt.savefig(FIG / "fitur_penting.png", dpi=120); plt.close()

cur = abstain_curve(yte, pred, probs.max(1))
cur.to_csv(D / "kurva_abstain.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(cur.coverage * 100, cur.accuracy * 100, marker="o", color="#6a1b9a")
ax.set_xlabel("Foto yang dijawab (%)"); ax.set_ylabel("Akurasi pada yang dijawab (%)")
ax.set_title("Random Forest: kalau ragu jangan dijawab"); ax.invert_xaxis(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "kurva_abstain.png", dpi=130); plt.close()

wk, best = worst_classes(yte, pred, labels, 15)
wk.to_csv(D / "kelas_terburuk.csv", index=False)

tbl_abs = "\n".join(f"| {r.coverage:.0%} | {r.n} | {r.accuracy:.1%} |" for r in cur.itertuples())
tbl_worst = "\n".join(f"| {r.spesies} | {r.recall:.2f} | {r.n} |" for r in wk.itertuples())
sections = [
    ("Ringkasan", f"""| Metrik | Nilai |
|---|---|
| Top-1 | **{top1:.1%}** |
| Top-3 | **{top3:.1%}** |
| Macro-F1 | {macro:.1%} |
| Waktu fit | {train_sec/60:.1f} menit |
| Jumlah pohon | 300 |"""),
    ("Keamanan", f"""| Metrik | Nilai |
|---|---|
| Foto beracun di test | {safe['n_test_beracun']} |
| Dikenali sebagai beracun | {safe['recall_beracun_sebagai_beracun']:.1%} |
| **Kesalahan fatal** | **{safe['fatal_n']} = {safe['fatal_rate']:.1%}** |
| Recall pangan | {safe['recall_pangan']:.1%} |
| Alarm beracun palsu pada pangan | {safe['alarm_palsu_beracun_pada_pangan']:.1%} |"""),
    ("Kalau ragu jangan dijawab", f"""| Cakupan | n | Akurasi |
|---|---|---|
{tbl_abs}"""),
    ("15 kelas terburuk", f"""| Spesies | Recall | n |
|---|---|---|
{tbl_worst}"""),
    ("Catatan", """Random Forest memakai embedding dari model 1.
Kelebihannya: tahan noise label. Kekurangannya: biasanya kalah akurat dari CNN
karena tidak bisa memakai struktur spasial gambar."""),
]
write_md(D / "HASIL.md", "Model 3 - Random Forest 300", sections)
print("DONE MODEL 3", flush=True)

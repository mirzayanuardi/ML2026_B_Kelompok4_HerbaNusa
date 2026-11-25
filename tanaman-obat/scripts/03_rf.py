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
                    save_json, show_fig, tampil, topk_acc, worst_classes)

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

# ukuran forest: diukur, bukan diperkirakan
n_nodes = int(sum(t.tree_.node_count for t in rf.estimators_))
n_leaves = int(sum(int((t.tree_.children_left == -1).sum()) for t in rf.estimators_))
# sklearn menyimpan tabel 100 kelas di SETIAP node (float64) -> besar sekali
size_sklearn_mb = sum(
    t.tree_.value.nbytes + sum(
        getattr(t.tree_, a).nbytes for a in
        ("children_left", "children_right", "feature", "threshold",
         "impurity", "n_node_samples", "weighted_n_node_samples"))
    for t in rf.estimators_) / 1024 ** 2
# format padat: 14 byte/node + distribusi kelas hanya di leaf (float32)
size_padat_mb = (n_nodes * 14 + n_leaves * N * 4) / 1024 ** 2

np.save(D / "probabilitas_test.npy", probs)
np.save(D / "pred_test.npy", pred)

tags = load_tags()
safe, fatal = safety_metrics(yte, pred, labels, tags)
metrics = {"model": "Random Forest 300", "top1": float(top1), "top3": float(top3),
           "macro_f1": float(macro), "train_sec": round(train_sec, 1),
           "n_nodes": n_nodes, "n_leaves": n_leaves,
           "size_sklearn_mb": round(size_sklearn_mb, 1),
           "size_padat_mb": round(size_padat_mb, 1), **safe}
save_json(D / "metrics.json", metrics)
print(f"TOPI-1 {top1:.4f} | TOP-3 {top3:.4f} | macroF1 {macro:.4f}", flush=True)
print(f"node {n_nodes:,} | sklearn {size_sklearn_mb:.0f} MB | padat {size_padat_mb:.0f} MB",
      flush=True)

cm = confusion_matrix(yte, pred, labels=list(range(N)))
fig, ax = plt.subplots(figsize=(12, 11))
ax.imshow(cm, cmap="cividis")
ax.set_title("Confusion matrix Random Forest (100 kelas)")
ax.set_xlabel("prediksi"); ax.set_ylabel("aktual")
show_fig(fig, FIG / "confusion_matrix.png", dpi=110)

per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(per_class)), per_class[np.argsort(per_class)], color="#6a1b9a")
ax.set_title("Recall per kelas (urut naik) - Random Forest"); ax.grid(axis="y", alpha=.3)
show_fig(fig, FIG / "recall_per_kelas.png", dpi=120)

imp = pd.Series(rf.feature_importances_).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(30), imp.values[:30], color="#6a1b9a")
ax.set_title("30 fitur embedding paling penting"); ax.set_xlabel("peringkat fitur")
show_fig(fig, FIG / "fitur_penting.png", dpi=120)

cur = abstain_curve(yte, pred, probs.max(1))
cur.to_csv(D / "kurva_abstain.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(cur.coverage * 100, cur.accuracy * 100, marker="o", color="#6a1b9a")
ax.set_xlabel("Foto yang dijawab (%)"); ax.set_ylabel("Akurasi pada yang dijawab (%)")
ax.set_title("Random Forest: kalau ragu jangan dijawab"); ax.invert_xaxis(); ax.grid(alpha=.3)
show_fig(fig, FIG / "kurva_abstain.png", dpi=130)

wk, best = worst_classes(yte, pred, labels, 15)
wk.to_csv(D / "kelas_terburuk.csv", index=False)

# --- tabel ringkasan ---
ringkasan = pd.DataFrame({
    "Metrik": ["Top-1", "Top-3", "Macro-F1", "Waktu fit", "Jumlah pohon",
               "Jumlah node", "Ukuran (scikit-learn)", "Ukuran (format padat)"],
    "Nilai": [f"{top1:.1%}", f"{top3:.1%}", f"{macro:.1%}", f"{train_sec/60:.1f} menit", "300",
              f"{n_nodes:,} ({n_leaves:,} leaf)", f"{size_sklearn_mb:.0f} MB",
              f"{size_padat_mb:.0f} MB"],
})
tampil("Ringkasan model 3", ringkasan)

# --- tabel keamanan ---
keamanan = pd.DataFrame({
    "Metrik": ["Foto beracun di test", "Dikenali sebagai beracun",
               "Kesalahan fatal", "Recall pangan", "Alarm beracun palsu pada pangan"],
    "Nilai": [safe['n_test_beracun'], f"{safe['recall_beracun_sebagai_beracun']:.1%}",
              f"{safe['fatal_n']} = {safe['fatal_rate']:.1%}", f"{safe['recall_pangan']:.1%}",
              f"{safe['alarm_palsu_beracun_pada_pangan']:.1%}"],
})
tampil("Keamanan", keamanan)

# --- tabel kurva abstain ---
tampil("Akurasi vs cakupan", cur.rename(columns={"coverage": "cakupan",
                                                 "accuracy": "akurasi", "n": "jumlah"}))

# --- tabel kelas terburuk ---
tampil("15 kelas terburuk", wk)
print("DONE MODEL 3", flush=True)

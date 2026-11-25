"""Tahap 5: perbandingan ketiga model + kurva abstain gabungan."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (OUT, load_labels, load_tags, safety_metrics, save_json,  # noqa: E402
                    show_fig, tampil)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

D = OUT / "04_perbandingan"
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)

labels = load_labels()
tags = load_tags()
yte = np.load(OUT / "01_cnn" / "y_test.npy")

# cakupan dari aturan penolakan (kalau sudah ada) supaya semua model bisa
# dibandingkan pada jumlah foto yang dijawab sama
_rules_f = OUT / "05_bahan_aplikasi" / "safety_rules.json"
RULES = (json.loads(_rules_f.read_text(encoding="utf-8")) if _rules_f.exists() else None)
COV = RULES["test_pada_ambang_ini"]["cakupan"] if RULES else None

models = {
    "CNN MobileNetV2": ("01_cnn", "probabilitas_test.npy"),
    "SVM RBF": ("02_svm", "probabilitas_test.npy"),
    "Random Forest 300": ("03_rf", "probabilitas_test.npy"),
}

rows, curves, percat = [], {}, []
fatal_rows = []
for name, (folder, fname) in models.items():
    p = np.load(OUT / folder / fname)
    pred = p.argmax(1)
    conf = p.max(1)
    safe, fatal = safety_metrics(yte, pred, labels, tags)
    acc_cov = None
    if COV:
        k = int(round(COV * len(yte)))
        sel = np.argsort(-conf)[:k]
        acc_cov = float((pred[sel] == yte[sel]).mean())
    rows.append({
        "model": name,
        "top1": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "fatal_rate": safe["fatal_rate"],
        "fatal_n": safe["fatal_n"],
        "recall_beracun": safe["recall_beracun_sebagai_beracun"],
        "false_alarm_beracun": safe["alarm_palsu_beracun_pada_pangan"],
        "akurasi_pada_cakupan": acc_cov,
    })
    percat.append({"model": name, **{k: safe[k] for k in
                                     ("recall_pangan", "recall_beracun_sebagai_beracun",
                                      "alarm_palsu_beracun_pada_pangan", "fatal_rate")}})
    cur = pd.read_csv(OUT / folder / "kurva_abstain.csv")
    curves[name] = cur
    for lbl, pr in zip(labels, (pred == yte)):
        pass
    # kesalahan fatal rinci
    tox = np.array([bool(tags.loc[l, "beracun"]) for l in labels])
    food = np.array([bool(tags.loc[l, "pangan"]) for l in labels])
    for i in np.where(fatal)[0]:
        fatal_rows.append({"model": name,
                           "aktual": labels[yte[i]],
                           "ditebak": labels[pred[i]]})

comp = pd.DataFrame(rows)
# top-3 dihitung ulang seragam
t3 = []
for name, (folder, fname) in models.items():
    p = np.load(OUT / folder / fname)
    top = np.argsort(-p, axis=1)[:, :3]
    t3.append(float(np.mean([yte[i] in top[i] for i in range(len(yte))])))
comp["top3"] = t3
comp = comp[["model", "top1", "top3", "macro_f1", "recall_beracun",
             "fatal_rate", "fatal_n", "false_alarm_beracun",
             "akurasi_pada_cakupan"]]
comp.to_csv(D / "tabel_model.csv", index=False)
pd.DataFrame(percat).to_csv(D / "per_kategori.csv", index=False)
pd.DataFrame(fatal_rows).to_csv(D / "kesalahan_fatal.csv", index=False)

# --- figur: bar akurasi ---
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(comp))
ax.bar(x - 0.2, comp.top1 * 100, width=0.38, label="Top-1", color="#1565c0")
ax.bar(x + 0.2, comp.top3 * 100, width=0.38, label="Top-3", color="#66bb6a")
for i, (a, b) in enumerate(zip(comp.top1, comp.top3)):
    ax.text(i - 0.2, a * 100 + 0.6, f"{a*100:.1f}", ha="center", fontsize=9)
    ax.text(i + 0.2, b * 100 + 0.6, f"{b*100:.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(comp.model, fontsize=9)
ax.set_ylabel("Akurasi (%)"); ax.set_ylim(0, 113)
ax.set_title("Perbandingan akurasi 3 algoritma")
ax.legend(loc="upper center", ncol=2, frameon=False)
ax.grid(axis="y", alpha=.3)
show_fig(fig, FIG / "bar_akurasi.png", dpi=130)

# --- kurva abstain gabungan ---
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for name, cur in curves.items():
    ax.plot(cur.coverage * 100, cur.accuracy * 100, marker="o", label=name)
ax.set_xlabel("Foto yang dijawab (%)"); ax.set_ylabel("Akurasi pada yang dijawab (%)")
ax.set_title("Kalau model ragu, jangan dijawab")
ax.invert_xaxis(); ax.grid(alpha=.3); ax.legend(fontsize=8)
show_fig(fig, FIG / "kurva_abstain_semua.png", dpi=130)

# --- figur fatal ---
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(comp.model, comp.fatal_rate * 100, color="#c62828")
for i, v in enumerate(comp.fatal_rate * 100):
    ax.text(i, v + 0.15, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Kesalahan fatal (%)")
ax.set_ylim(0, float(comp.fatal_rate.max()) * 100 * 1.35)
ax.set_title("Beracun ditebak sebagai tanaman pangan")
ax.grid(axis="y", alpha=.3)
show_fig(fig, FIG / "kesalahan_fatal.png", dpi=130)

cur_all = pd.concat([c.assign(model=k) for k, c in curves.items()])
cur_all.to_csv(D / "kurva_abstain.csv", index=False)

best = comp.sort_values("top1", ascending=False).iloc[0]
row = {r.model: r for r in comp.itertuples()}
cnn = row["CNN MobileNetV2"]
svm = row["SVM RBF"]
rfm = row["Random Forest 300"]
selisih = (svm.top1 - cnn.top1) * 100


def idn(v, d=1):
    return f"{v:.{d}f}".replace(".", ",")


# angka ambang + ukuran diambil dari artefak, bukan ditulis tangan
rules_f = OUT / "05_bahan_aplikasi" / "safety_rules.json"
meta_f = OUT / "05_bahan_aplikasi" / "model_meta.json"
if rules_f.exists():
    _r = json.loads(rules_f.read_text(encoding="utf-8"))
    ambang = _r["ambang_keyakinan"]
    tst = _r["test_pada_ambang_ini"]
    kal_ambang = (f"Dengan `AMBANG = {idn(ambang, 2)}`, "
                  f"{idn(tst['cakupan'] * 100)}% foto dijawab "
                  f"dengan akurasi {idn(tst['akurasi_pada_yang_dijawab'] * 100)}%")
else:
    kal_ambang = "Angka ambang belum dihitung (jalankan `06_threshold.py`)"
ukuran_tflite = (f"{idn(json.loads(meta_f.read_text(encoding='utf-8'))['size_mb'], 2)} MB"
                 if meta_f.exists() else "4,6 MB")
_rf = OUT / "03_rf" / "metrics.json"
if _rf.exists():
    _m = json.loads(_rf.read_text(encoding="utf-8"))
    ukuran_rf = f"{_m.get('size_padat_mb', 175):.0f} MB".replace(".", ",")
else:
    ukuran_rf = "175 MB"

# perbandingan adil: pada cakupan yang sama, siapa paling akurat?
if COV and cnn.akurasi_pada_cakupan is not None:
    kal_cov = (f"Pada cakupan yang sama ({idn(COV * 100)}% foto dijawab), CNN justru "
               f"**paling akurat**: {idn(cnn.akurasi_pada_cakupan * 100)}% "
               f"vs SVM {idn(svm.akurasi_pada_cakupan * 100)}% "
               f"dan RF {idn(rfm.akurasi_pada_cakupan * 100)}%.")
    kal_menang = ("CNN menang setelah aturan penolakan dipakai, jadi SVM tidak "
                  "perlu dibawa ke HP sama sekali.")
    head_cov = f"Akurasi @ cakupan {idn(COV * 100)}%"
    acc_cnn = f"{idn(cnn.akurasi_pada_cakupan * 100)}%"
    acc_svm = f"{idn(svm.akurasi_pada_cakupan * 100)}%"
    acc_rfm = f"{idn(rfm.akurasi_pada_cakupan * 100)}%"
else:
    kal_cov = ("Perbandingan pada cakupan sama belum bisa dihitung karena "
               "`safety_rules.json` belum ada.")
    kal_menang = ""
    head_cov = "Akurasi @ cakupan"
    acc_cnn = acc_svm = acc_rfm = "—"

# --- tabel perbandingan ---
comp_tampil = comp.copy()
for _c in ("top1", "top3", "macro_f1", "recall_beracun", "fatal_rate",
           "false_alarm_beracun", "akurasi_pada_cakupan"):
    comp_tampil[_c] = comp_tampil[_c].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
tampil("Perbandingan ketiga model di data test", comp_tampil)

# --- tabel pilihan deploy ---
pil = pd.DataFrame([
    ["CNN murni (`model.tflite`)", f"{idn(cnn.top1 * 100)}%", acc_cnn, ukuran_tflite,
     "rendah, 1 inferensi"],
    ["CNN + SVM", f"{idn(svm.top1 * 100)}%", acc_svm, "belasan MB",
     "tinggi, kernel RBF manual"],
    ["CNN + Random Forest", f"{idn(rfm.top1 * 100)}%", acc_rfm, f"+/- {ukuran_rf}",
     "sangat tinggi, tidak layak"],
], columns=["Pilihan", "Top-1 semua foto", head_cov, "Ukuran tambahan", "Kerumitan di aplikasi"])
tampil("Pilihan yang bisa dikirim ke HP", pil)

# --- contoh kesalahan fatal ---
if fatal_rows:
    tampil("Contoh kesalahan fatal", pd.DataFrame(fatal_rows).head(15))

save_json(D / "ringkasan.json", {"tabel": comp.to_dict("records"), "pemenang": best.model})
print("DONE TAHAP 5", flush=True)

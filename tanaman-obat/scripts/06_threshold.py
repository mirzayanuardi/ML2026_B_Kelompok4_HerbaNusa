"""Tahap 6a: tentukan AMBANG penolakan dari data VALIDASI.

Aturan: pilih ambang confidence terendah yang masih memberi akurasi
>= TARGET pada foto yang dijawab. Memakai val, bukan test, supaya test
tetap bersih sebagai laporan akhir.

Output: output/05_bahan_aplikasi/safety_rules.json + figures/ambang.png
        output/01_cnn/kurva_ambang_val.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, load_tags, save_json

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = OUT / "01_cnn"
D = OUT / "05_bahan_aplikasi"
FIG = D / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TARGET = 0.93          # akurasi yang diinginkan pada foto yang dijawab
MIN_COVERAGE = 0.40    # jangan sampai aplikasi menolak > 60% foto

labels = json.loads((OUT / "_data" / "labels.json").read_text(encoding="utf-8"))
prob_va = np.load(SRC / "probabilitas_val.npy")
y_va = np.load(SRC / "y_val.npy")
conf = prob_va.max(1)
pred = prob_va.argmax(1)
correct = (pred == y_va)

print(f"validasi: {len(y_va)} gambar | akurasi penuh {(correct).mean():.4f}", flush=True)


def stat(thr):
    sel = conf >= thr
    n = int(sel.sum())
    if n == 0:
        return {"ambang": float(thr), "n_dijawab": 0, "cakupan": 0.0, "akurasi": None}
    return {"ambang": float(thr), "n_dijawab": n, "cakupan": float(sel.mean()),
            "akurasi": float(correct[sel].mean())}


grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
rows = [stat(t) for t in grid]
df = pd.DataFrame(rows)
df.to_csv(SRC / "kurva_ambang_val.csv", index=False)

ok = df[(df.akurasi.notna()) & (df.akurasi >= TARGET) & (df.cakupan >= MIN_COVERAGE)]
if len(ok):
    chosen = ok.sort_values("ambang").iloc[0]          # ambang terendah yang memenuhi
    note = f"ambang terendah yang memberi akurasi >= {TARGET:.0%} pada foto yang dijawab"
else:
    # tidak ada yang memenuhi -> pakai ambang dengan akurasi tertinggi, cakupan >= minimum
    cand = df[df.cakupan >= MIN_COVERAGE]
    chosen = cand.sort_values("akurasi", ascending=False).iloc[0]
    note = (f"tidak ada ambang yang mencapai {TARGET:.0%} dengan cakupan >= {MIN_COVERAGE:.0%}; "
            "dipakai ambang dengan akurasi terbaik")

AMBANG = float(chosen.ambang)
print(f"\nAMBANG = {AMBANG:.2f} | cakupan {chosen.cakupan:.1%} | "
      f"akurasi pada yang dijawab {chosen.akurasi:.1%}", flush=True)

# --- evaluasi ambang ini di TEST (sekali, hanya untuk laporan) ---
prob_te = np.load(SRC / "probabilitas_test.npy")
y_te = np.load(SRC / "y_test.npy")
conf_te = prob_te.max(1)
pred_te = prob_te.argmax(1)
sel = conf_te >= AMBANG
acc_te = float((pred_te[sel] == y_te[sel]).mean())

tags = load_tags()
tox = np.array([bool(tags.loc[l, "beracun"]) for l in labels])
food = np.array([bool(tags.loc[l, "pangan"]) for l in labels])
fatal_all = tox[y_te] & (pred_te != y_te) & food[pred_te] & ~tox[pred_te]
fatal_sel = fatal_all & sel
n_tox_sel = int((tox[y_te] & sel).sum())

print(f"TEST pada ambang ini: dijawab {sel.mean():.1%} | akurasi {acc_te:.1%} | "
      f"fatal {int(fatal_sel.sum())}/{n_tox_sel}", flush=True)

rules = {
    "ambang_keyakinan": round(AMBANG, 2),
    "aturan": "tolak menjawab jika probabilitas tertinggi < ambang_keyakinan",
    "dasar_penentuan": note,
    "dipilih_dari": "data validasi (1.476 gambar), bukan data test",
    "target_akurasi_pada_yang_dijawab": TARGET,
    "cakupan_minimum": MIN_COVERAGE,
    "validasi": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in chosen.to_dict().items()},
    "test_pada_ambang_ini": {
        "n_test": int(len(y_te)),
        "cakupan": round(float(sel.mean()), 4),
        "akurasi_pada_yang_dijawab": round(acc_te, 4),
        "akurasi_tanpa_penolakan": round(float((pred_te == y_te).mean()), 4),
        "fatal_n": int(fatal_sel.sum()),
        "fatal_rate_pada_beracun_yang_dijawab": round(float(fatal_sel.sum() / max(n_tox_sel, 1)), 4),
        "beracun_dijawab": n_tox_sel,
    },
    "aturan_margin": {
        "dipakai": False,
        "alasan": "sudah diuji pada model ini: aturan margin tidak mengubah cakupan maupun akurasi",
    },
    "larangan": [
        "jangan menurunkan ambang agar pengguna tidak kecewa",
        "jangan memakai satu jawaban saja; selalu tampilkan top-3",
        "jika salah satu dari top-3 punya level_racun >= 2, tampilkan peringatan merah",
    ],
}
save_json(D / "safety_rules.json", rules)

# --- figur ---
fig, ax = plt.subplots(figsize=(7.2, 4.6))
d = df[df.akurasi.notna()]
ax.plot(d.cakupan * 100, d.akurasi * 100, color="#1565c0", lw=1.6)
ax.axhline(TARGET * 100, ls="--", color="#c62828", lw=1, label=f"target {TARGET:.0%}")
ax.scatter([chosen.cakupan * 100], [chosen.akurasi * 100], s=70, color="#c62828", zorder=5,
           label=f"AMBANG {AMBANG:.2f}")
ax.set_xlabel("Foto yang dijawab (%)"); ax.set_ylabel("Akurasi pada yang dijawab (%)")
ax.set_title("Pemilihan AMBANG dari data validasi")
ax.invert_xaxis(); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "ambang.png", dpi=130); plt.close()

print("DONE TAHAP 6a", flush=True)

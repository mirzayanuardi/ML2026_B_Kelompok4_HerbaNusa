"""Utilitas bersama untuk pipeline tanaman obat (tim ML).

Semua output masuk ke  tanaman-obat/output/
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\Wahyu Pratomo\Project Machine Learning\tanaman-obat")
OUT = BASE / "output"
DATA = OUT / "_data"
TAGS_CSV = BASE / "analisis" / "peta_tag_draf.csv"

for _d in (OUT, DATA):
    _d.mkdir(parents=True, exist_ok=True)


def load_labels():
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))


def load_tags():
    df = pd.read_csv(TAGS_CSV)
    df["folder"] = df["folder"].astype(str)
    return df.set_index("folder")


def tag_arrays(labels, tags):
    tox = np.array([bool(tags.loc[l, "beracun"]) for l in labels])
    food = np.array([bool(tags.loc[l, "pangan"]) for l in labels])
    med = np.array([bool(tags.loc[l, "obat"]) for l in labels])
    return tox, food, med


def topk_acc(y, prob, k=3):
    top = np.argsort(-prob, axis=1)[:, :k]
    return float(np.mean([y[i] in top[i] for i in range(len(y))]))


def safety_metrics(y, pred, labels, tags):
    tox, food, _ = tag_arrays(labels, tags)
    tt = tox[y]
    fatal = tt & (pred != y) & food[pred] & ~tox[pred]
    out = {
        "n_test": int(len(y)),
        "n_test_beracun": int(tt.sum()),
        "recall_beracun_sebagai_beracun": float(tox[pred][tt].mean()) if tt.sum() else None,
        "fatal_n": int(fatal.sum()),
        "fatal_rate": float(fatal[tt].mean()) if tt.sum() else None,
        "recall_pangan": float(food[pred][food[y]].mean()),
        "alarm_palsu_beracun_pada_pangan": float(tox[pred][food[y]].mean()),
    }
    return out, fatal


def abstain_curve(y, pred, conf):
    order = np.argsort(-conf)
    rows = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        k = max(1, int(round(cov * len(pred))))
        sel = order[:k]
        rows.append({"coverage": cov, "accuracy": float((y[sel] == pred[sel]).mean()), "n": k})
    return pd.DataFrame(rows)


def worst_classes(y, pred, labels, n=15):
    rows = []
    for i, lab in enumerate(labels):
        m = y == i
        if m.sum():
            rows.append({"spesies": lab, "recall": float((pred[m] == i).mean()), "n": int(m.sum())})
    df = pd.DataFrame(rows).sort_values("recall")
    return df.head(n), df.tail(min(10, len(df)))


def show_fig(fig, path, dpi=130):
    """Simpan grafik ke PNG sekaligus tampilkan inline saat dijalankan di Jupyter."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        from IPython.display import Image, display
        display(Image(filename=str(path)))
    except Exception:
        pass
    import matplotlib.pyplot as plt
    plt.close(fig)


def tampil(judul, tabel):
    """Tampilkan DataFrame sebagai tabel di notebook (jatuh ke print di terminal)."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"**{judul}**"))
        display(tabel)
    except Exception:
        print(f"\n{judul}\n{tabel}")


def tampil_teks(md):
    """Tampilkan teks markdown di notebook."""
    try:
        from IPython.display import Markdown, display
        display(Markdown(md))
    except Exception:
        print(md)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  json ->", path)

"""Builds HerbaNusa's bundled assets from the ML handoff in ../tanaman-obat.

Run from anywhere:  python tool/build_assets.py
Needs numpy, Pillow and TensorFlow (the same env that trained the model).

Writes:
  assets/model/model.tflite, labels.json, model_card.json
  assets/data/knowledge_base.json
  assets/plants/<slug>.jpg      one dataset photo per species
  assets/images/sample_plant.jpg
  test/fixtures/sample_<i>.jpg + .bin   TF preprocessing reference tensors
  android launcher icons (leaf)
"""

import csv
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
ML = ROOT / "tanaman-obat"
HANDOFF = ML / "output" / "05_bahan_aplikasi"
DATASET_ZIP = ROOT / "Indonesia-Medicinal-Plant-dataset.zip"
ZIP_PREFIX = "Indonesia-Medicinal-Plant-Dataset"

# Toxicity level per species flagged `beracun` in peta_tag_draf.csv. The CSV has
# no level, so this is the app team's reading of its notes:
#   3 = whole plant or the usual part is dangerous      -> DILARANG
#   2 = dangerous unless processed / only under supervision
#   1 = only a specific part or a large dose is a problem
# shortcut: hand-assigned from the CSV notes, have the ML team confirm before release.
LEVEL = {
    "Caladium Cicolor (Keladi)": 3,
    "Catharanthus Roseus (Tapak Dara)": 3,
    "Cestrum Nocturnum (Sedap Malam)": 3,
    "Crinum Asiaticum (Bakung Putih)": 3,
    "Derris Elliptica (Tuba)": 3,
    "Gynura Segetum (Daun Dewa)": 3,
    "Jatropa Multifida (Betadin)": 3,
    "Melia Azedarach (Mindi)": 3,
    "Phytolacca Americana (Mrico Kepyar)": 3,
    "Ricinus Communis (Jarak)": 3,
    "Zephyranthes Candida (Lili Hujan Putih)": 3,
    "Acorus Calamus (Dlingo)": 2,
    "Dioscorea Hispida (Gadung)": 2,
    "Euphorbia Tirucalli (Patah Tulang)": 2,
    "Isotoma Longiflora (Ki Tolod)": 2,
    "Phaleria Macrocarpa (Mahkota Dewa)": 2,
    "Plumbago Zeylanica (Daun Encok)": 2,
    "Raulvolvia Serpentina (Pule Pandak)": 2,
    "Ruta Angustifolia (Inggu)": 2,
    "Typhonium Flagelliforme (Keladi Tikus)": 2,
    "Aloe Vera (Lidah Buaya)": 1,
    "Alstonia Scholaris (Pulai)": 1,
    "Annona Muricata (Sirsak)": 1,
    "Annona Squamosa (Srikaya)": 1,
    "Averrhoa Bilimbi (Belimbing Wuluh)": 1,
    "Melaleuca Leucadendra (Kayu Putih)": 1,
    "Mirabilis Jalapa (Bunga Pukul Empat)": 1,
    "Prunus Domestica (Plum)": 1,
    "Pyrus Communis (Pir)": 1,
    "Tinospora Tuberculata (Brotowali)": 1,
}

# Pairs measured on the test set after the 0.27 threshold (README section 3.6):
# predicted species -> poisonous species the photo actually showed.
CONFUSED = [
    ("Piper Sarmentosum (Cabean)", "Gynura Segetum (Daun Dewa)", 0.65),
    ("Pluchea Indica (Beluntas)", "Gynura Segetum (Daun Dewa)", 0.57),
    ("Zingiber Officinale (Jahe)", "Acorus Calamus (Dlingo)", 0.56),
    ("Zingiber Officinale (Jahe)", "Melia Azedarach (Mindi)", 0.37),
    ("Ocimum Basilicum (Kemangi)", "Cestrum Nocturnum (Sedap Malam)", 0.36),
    ("Syzygium Polyanthum (Salam)", "Gynura Segetum (Daun Dewa)", 0.30),
    ("Melissa officinalis (Lemon Balm)", "Gynura Segetum (Daun Dewa)", 0.28),
    ("Ocimum Basilicum (Kemangi)", "Phytolacca Americana (Mrico Kepyar)", 0.27),
]


def slug(name):
    return re.sub("[^a-z0-9]+", "_", name.lower()).strip("_")


def first_aid(row, level):
    """General first-aid steps chosen from the exposure route in the notes."""
    if level == 0:
        return [
            "Basis pengetahuan tidak mencatat racun pada tanaman ini.",
            "Hentikan pemakaian bila muncul gatal, ruam, bengkak, atau sesak napas (tanda alergi) dan periksakan diri bila tidak membaik.",
        ]
    danger = f"{row['bagian_bahaya']} {row['catatan']}".lower()
    steps = []
    if level >= 2:
        steps.append(
            "Jika tertelan: jangan memancing muntah. Kumur dengan air bersih, lalu segera ke fasilitas kesehatan terdekat atau hubungi 119."
        )
    else:
        steps.append(
            "Jika bagian berbahaya tertelan atau muncul mual, muntah, diare, atau pusing: hentikan konsumsi dan periksakan diri ke fasilitas kesehatan."
        )
    if "oksalat" in danger:
        steps.append(
            "Rasa terbakar atau bengkak di mulut: kumur dan minum air dingin sedikit-sedikit. Bibir, lidah, atau tenggorokan membengkak atau sulit bernapas: segera ke IGD."
        )
    if any(k in danger for k in ("getah", "lateks", "minyak", "fototoksik", "iritan")):
        steps.append(
            "Getah atau minyak mengenai kulit: cuci dengan sabun dan air mengalir minimal 15 menit, hindari sinar matahari langsung pada bagian itu."
        )
        steps.append(
            "Mengenai mata: bilas dengan air mengalir 15 menit tanpa dikucek, lalu periksakan ke dokter."
        )
    steps.append("Bawa sampel atau foto tanaman saat ke fasilitas kesehatan.")
    return steps


def build_knowledge_base(labels):
    with open(ML / "analisis" / "peta_tag_draf.csv", encoding="utf-8") as f:
        rows = {row["folder"]: row for row in csv.DictReader(f)}
    assert set(rows) == set(labels), set(rows) ^ set(labels)
    toxic = {name for name, row in rows.items() if row["beracun"] == "1"}
    assert toxic == set(LEVEL), toxic ^ set(LEVEL)

    confused = {}
    for predicted, actual, conf in CONFUSED:
        assert predicted in rows and actual in rows, (predicted, actual)
        confused.setdefault(predicted, []).append({"nama": actual, "keyakinan": conf})

    kb = {}
    for name in labels:
        row = rows[name]
        scientific, local = re.match(r"(.+?) \((.+)\)$", name).groups()
        level = LEVEL.get(name, 0)
        kb[name] = {
            "ilmiah": scientific,
            "lokal": local,
            "pangan": int(row["pangan"]),
            "obat": int(row["obat"]),
            "beracun": int(row["beracun"]),
            "level_racun": level,
            "bagian_aman": row["bagian_aman"],
            "bagian_bahaya": row["bagian_bahaya"],
            "catatan": row["catatan"],
            "p3k": first_aid(row, level),
            "tertukar_dengan": confused.get(name, []),
        }
    return kb


def build_model_card(labels):
    out = ML / "output" / "01_cnn"
    probs = np.load(out / "probabilitas_test.npy")
    y = np.load(out / "y_test.npy")
    n_train = len(np.load(out / "y_train.npy"))
    rules = json.loads((HANDOFF / "safety_rules.json").read_text(encoding="utf-8"))
    metrics = json.loads((HANDOFF / "metrics_model_1.json").read_text(encoding="utf-8"))
    pred = probs.argmax(1)
    assert abs((pred == y).mean() - metrics["top1"]) < 1e-9, "npy does not match shipped model"
    threshold = rules["ambang_keyakinan"]
    answered = probs.max(1) >= threshold
    per_class = {}
    for i, name in enumerate(labels):
        mask = y == i
        tp = int(((pred == i) & mask).sum())
        predicted = int((pred == i).sum())
        recall = tp / mask.sum()
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
        per_class[name] = {"recall": round(float(recall), 4), "f1": round(float(f1), 4), "support": int(mask.sum())}
    return {
        "threshold": threshold,
        "top1": metrics["top1"],
        "top3": metrics["top3"],
        "macro_f1": metrics["macro_f1"],
        "test_images": int(len(y)),
        "train_images": n_train,
        "coverage": float(answered.mean()),
        "accuracy_answered": float((pred[answered] == y[answered]).mean()),
        "fatal_rate": metrics["fatal_rate"],
        "fatal_rate_answered": rules["test_pada_ambang_ini"]["fatal_rate_pada_beracun_yang_dijawab"],
        "per_class": per_class,
    }


def tf_reference(jpg_bytes):
    """Exactly the training/export preprocessing (05_export_tflite.load_raw)."""
    import tensorflow as tf

    img = tf.io.decode_image(jpg_bytes, channels=3, expand_animations=False)
    return tf.image.resize(img, [224, 224]).numpy().astype(np.float32)


def export_images_and_fixtures(labels):
    import tensorflow as tf

    test = {}
    with open(ML / "output" / "_data" / "test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test.setdefault(row["label"], []).append(Path(row["path"]).name)

    interp = tf.lite.Interpreter(model_path=str(HANDOFF / "model.tflite"))
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]

    def predict(x):
        interp.set_tensor(inp, x[None])
        interp.invoke()
        return interp.get_tensor(outp)[0]

    plants = APP / "assets" / "plants"
    shutil.rmtree(plants, ignore_errors=True)
    plants.mkdir(parents=True)
    fixtures = APP / "test" / "fixtures"
    shutil.rmtree(fixtures, ignore_errors=True)
    fixtures.mkdir(parents=True)

    with zipfile.ZipFile(DATASET_ZIP) as z:
        def read(label, file):
            return z.read(f"{ZIP_PREFIX}/{label}/{file}")

        for name in labels:
            # Catalog photo: the test image the model recognises most confidently.
            best = max(test[name], key=lambda f: predict(tf_reference(read(name, f)))[labels.index(name)])
            (plants / f"{slug(name)}.jpg").write_bytes(read(name, best))

        sample = "Zingiber Officinale (Jahe)"  # has a measured poisonous look-alike, so the demo shows that note
        (APP / "assets" / "images" / "sample_plant.jpg").write_bytes((plants / f"{slug(sample)}.jpg").read_bytes())

        expected = {}
        for i, name in enumerate(["Ricinus Communis (Jarak)", "Piper Betle (Sirih)", "Pyrus Communis (Pir)", "Dioscorea Hispida (Gadung)"]):
            data = read(name, test[name][0])
            (fixtures / f"sample_{i}.jpg").write_bytes(data)
            tensor = tf_reference(data)
            (fixtures / f"sample_{i}.bin").write_bytes(tensor.tobytes())
            expected[f"sample_{i}"] = labels[int(predict(tensor).argmax())]
        (fixtures / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")


def launcher_icons():
    """Leaf on a green tile, replacing the cat icon inherited from MeowScan."""
    res = APP / "android" / "app" / "src" / "main" / "res"
    sizes = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}

    def leaf(size, scale):
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c, r = size / 2, size * scale
        d.chord([c - r, c - r * 1.15, c + r * 1.4, c + r * 1.25], 120, 330, fill="white")
        d.chord([c - r * 1.4, c - r * 1.25, c + r, c + r * 1.15], 300, 150, fill="white")
        d.line([c - r * 0.75, c + r * 0.9, c + r * 0.6, c - r * 0.7], fill=(31, 107, 66), width=max(2, size // 40))
        return img

    for density, px in sizes.items():
        folder = res / f"mipmap-{density}"
        tile = Image.new("RGBA", (px, px), (0, 0, 0, 0))
        ImageDraw.Draw(tile).rounded_rectangle([0, 0, px - 1, px - 1], radius=px // 4, fill=(47, 143, 91))
        tile.alpha_composite(leaf(px, 0.26))
        tile.save(folder / "ic_launcher.png")
        fg_px = px * 108 // 48
        leaf(fg_px, 0.17).save(folder / "ic_launcher_foreground.png")
        Image.new("RGBA", (fg_px, fg_px), (47, 143, 91, 255)).save(folder / "ic_launcher_background.png")
        for stale in folder.glob("ic_launcher_round.png"):
            stale.unlink()


def main():
    labels = json.loads((HANDOFF / "labels.json").read_text(encoding="utf-8"))
    model_dir = APP / "assets" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (APP / "assets" / "data").mkdir(parents=True, exist_ok=True)
    (APP / "assets" / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy(HANDOFF / "model.tflite", model_dir / "model.tflite")
    (model_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    (model_dir / "model_card.json").write_text(json.dumps(build_model_card(labels), indent=1), encoding="utf-8")
    kb = build_knowledge_base(labels)
    (APP / "assets" / "data" / "knowledge_base.json").write_text(
        json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    export_images_and_fixtures(labels)
    launcher_icons()
    print("ok:", len(labels), "species,", sum(v["beracun"] for v in kb.values()), "poisonous")


if __name__ == "__main__":
    main()

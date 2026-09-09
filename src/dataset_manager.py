"""UC7: Mengelola Dataset — masukkan citra berlabel ke folder processed + tabel `dataset`."""
import datetime as dt
import shutil
from pathlib import Path

import cv2

from . import db
from .face_detection import detect, crop
from .preprocessing import SIZE, praproses

LABELS = ("happy", "sad", "angry", "neutral")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tiff"}


def siapkan(citra):
    """Samakan citra ke grayscale 128x128: crop wajah kalau ketemu, kalau tidak pakai utuh."""
    box = detect(citra)
    if box is not None:
        return crop(citra, box)
    # ponytail: dataset publik (FER2013 48x48) sudah ter-crop wajah, Haar sering gagal di
    # citra sekecil itu. Pakai citra utuh — hasilnya sama saja karena isinya memang wajah.
    return praproses(citra)


# AffectNet format YOLO punya 8 kelas; 4 di luar scope skripsi (contempt/disgust/fear/surprise)
YOLO_KELAS = {0: "angry", 4: "happy", 5: "neutral", 6: "sad"}


def impor_yolo(folder, cap=500, kelas=YOLO_KELAS, tujuan=None):
    """Salin citra AffectNet (YOLO) ke `data/raw/<label>/`, maksimal `cap` per kelas.

    Kelas dibaca dari angka pertama file label. `cap` menyeimbangkan jumlah antar kelas
    sekaligus menahan waktu training SVM tetap masuk akal.
    """
    folder = Path(folder)
    tujuan = Path(tujuan or RAW)
    n = {v: 0 for v in kelas.values()}
    for f in sorted((folder / "labels").glob("*.txt")):
        isi = f.read_text().split()
        if not isi:
            continue
        label = kelas.get(int(isi[0]))
        if label is None or n[label] >= cap:
            continue
        citra = next((folder / "images").glob(f.stem + ".*"), None)
        if citra is None:
            continue
        (tujuan / label).mkdir(parents=True, exist_ok=True)
        shutil.copy2(citra, tujuan / label / citra.name)
        n[label] += 1
    return n


def ingest(folder, sumber_data, conn=None):
    """Baca `folder/<label>/*.jpg` -> simpan hasil preprocessing + catat ke tabel dataset.

    `sumber_data`: 'publik' (dataset unduhan) atau 'lapangan' (foto di hotel).
    Mengembalikan jumlah citra masuk per label.
    """
    conn = conn or db.connect()
    hasil = {}
    tanggal = dt.date.today().isoformat()
    for label in LABELS:
        src = Path(folder) / label
        if not src.is_dir():
            continue
        tujuan = PROCESSED / label
        tujuan.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.iterdir()):
            if f.suffix.lower() not in EXT:
                continue
            citra = cv2.imread(str(f))
            if citra is None:
                continue
            out = tujuan / (f.stem + ".png")
            cv2.imwrite(str(out), siapkan(citra))
            cur = conn.execute(
                "INSERT OR IGNORE INTO dataset"
                " (path_file,label_emosi,sumber_data,tanggal_input) VALUES (?,?,?,?)",
                (str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
                 label, sumber_data, tanggal))
            n += cur.rowcount           # 0 kalau citra ini sudah pernah tercatat
        hasil[label] = n
    conn.commit()
    return hasil


def ringkasan(conn=None):
    """Jumlah citra per label per sumber — dipakai halaman admin & BAB IV."""
    conn = conn or db.connect()
    return [dict(r) for r in conn.execute(
        "SELECT label_emosi, sumber_data, COUNT(*) AS jumlah FROM dataset"
        " GROUP BY label_emosi, sumber_data ORDER BY label_emosi")]


def hapus(id_citra, conn=None, hapus_file=True):
    """Hapus satu citra dari dataset (UC7). File ikut dibuang kecuali diminta sebaliknya."""
    conn = conn or db.connect()
    row = conn.execute("SELECT path_file FROM dataset WHERE id_citra=?",
                       (id_citra,)).fetchone()
    if row is None:
        return False
    if hapus_file:
        (ROOT / row["path_file"]).unlink(missing_ok=True)
    conn.execute("DELETE FROM dataset WHERE id_citra=?", (id_citra,))
    conn.commit()
    return True


if __name__ == "__main__":
    import tempfile
    import numpy as np

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for label in ("happy", "angry"):
            d = Path(tmp) / label
            d.mkdir()
            for i in range(3):
                cv2.imwrite(str(d / f"{label}{i}.png"),
                            rng.integers(0, 256, (48, 48), dtype=np.uint8))
        (Path(tmp) / "happy" / "catatan.txt").write_text("bukan citra")

        conn = db.connect(":memory:")
        PROCESSED = Path(tmp) / "out"
        n = ingest(tmp, "publik", conn)
        assert n == {"happy": 3, "angry": 3}, n          # file .txt tidak ikut terhitung
        assert len(ringkasan(conn)) == 2
        assert (PROCESSED / "happy" / "happy0.png").exists()
        assert cv2.imread(str(PROCESSED / "happy" / "happy0.png"),
                          cv2.IMREAD_GRAYSCALE).shape == (SIZE, SIZE)

        id_ = conn.execute("SELECT MIN(id_citra) i FROM dataset").fetchone()["i"]
        assert hapus(id_, conn, hapus_file=False) is True
        assert hapus(99999, conn) is False
    print("dataset_manager OK")

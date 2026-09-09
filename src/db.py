"""Koneksi SQLite + inisialisasi skema (BAB IV 4.4.3)."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "db" / "fer.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    id_citra INTEGER PRIMARY KEY AUTOINCREMENT,
    path_file TEXT NOT NULL,
    label_emosi TEXT NOT NULL CHECK(label_emosi IN ('happy','sad','angry','neutral')),
    sumber_data TEXT NOT NULL,
    tanggal_input TEXT NOT NULL
);

-- satu file citra hanya boleh tercatat sekali; menekan tombol ingest berulang
-- tidak boleh menggandakan dataset karena duplikat membocorkan data latih ke data uji
CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_path ON dataset(path_file);

CREATE TABLE IF NOT EXISTS model (
    id_model INTEGER PRIMARY KEY AUTOINCREMENT,
    path_model TEXT NOT NULL,
    akurasi REAL,
    tanggal_training TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS log_deteksi (
    id_log INTEGER PRIMARY KEY AUTOINCREMENT,
    waktu_deteksi TEXT NOT NULL,
    label_emosi TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    id_model INTEGER,
    FOREIGN KEY(id_model) REFERENCES model(id_model)
);
"""


def connect(path=DB_PATH):
    """Buka koneksi, buat skema kalau belum ada. Row hasil query bisa diakses by-name."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# Kode tampilan BAB IV 4.4.4 — kolom turunan, bukan primary key fisik.
def kode_citra(i):  return f"IMG-{i:05d}"
def kode_model(i):  return f"MDL-{i:03d}"
def kode_log(i, waktu):  return f"LOG-{waktu[:4]}{waktu[5:7]}{i:05d}"


if __name__ == "__main__":
    conn = connect(":memory:")
    tabel = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"dataset", "model", "log_deteksi"} <= tabel, tabel

    conn.execute("INSERT INTO dataset (path_file,label_emosi,sumber_data,tanggal_input)"
                 " VALUES ('a.jpg','happy','publik','2026-01-01')")
    try:
        conn.execute("INSERT INTO dataset (path_file,label_emosi,sumber_data,tanggal_input)"
                     " VALUES ('b.jpg','bingung','publik','2026-01-01')")
        raise SystemExit("FAIL: CHECK label_emosi tidak jalan")
    except sqlite3.IntegrityError:
        pass

    assert kode_citra(1) == "IMG-00001"
    assert kode_model(1) == "MDL-001"
    assert kode_log(1, "2026-10-05") == "LOG-20261000001"
    print("db.py OK")

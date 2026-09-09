"""UC9: Log Deteksi — simpan & baca hasil klasifikasi real-time ke tabel `log_deteksi`."""
import datetime as dt

from . import db


def catat(label, confidence, id_model=None, conn=None):
    """Simpan satu hasil deteksi. Mengembalikan id_log."""
    conn = conn or db.connect()
    if id_model is None:
        row = conn.execute("SELECT MAX(id_model) AS i FROM model").fetchone()
        id_model = row["i"]                 # None kalau belum pernah training — tetap boleh dicatat
    cur = conn.execute(
        "INSERT INTO log_deteksi (waktu_deteksi,label_emosi,confidence_score,id_model)"
        " VALUES (?,?,?,?)",
        (dt.datetime.now().isoformat(" ", "seconds"), label, float(confidence), id_model))
    conn.commit()
    return cur.lastrowid


def baca(batas=100, conn=None):
    """Log terbaru lebih dulu, siap ditampilkan di halaman admin."""
    conn = conn or db.connect()
    return [dict(r) for r in conn.execute(
        "SELECT * FROM log_deteksi ORDER BY id_log DESC LIMIT ?", (batas,))]


def rekap(conn=None):
    """Jumlah deteksi dan rata-rata confidence per emosi — bahan pembahasan BAB V."""
    conn = conn or db.connect()
    return [dict(r) for r in conn.execute(
        "SELECT label_emosi, COUNT(*) AS jumlah, AVG(confidence_score) AS rata_confidence"
        " FROM log_deteksi GROUP BY label_emosi ORDER BY jumlah DESC")]


if __name__ == "__main__":
    conn = db.connect(":memory:")
    conn.execute("INSERT INTO model (path_model,akurasi,tanggal_training)"
                 " VALUES ('m.joblib',0.7,'2026-01-01')")

    assert catat("happy", 0.91, conn=conn) == 1
    catat("happy", 0.71, conn=conn)
    catat("sad", 0.55, conn=conn)

    log = baca(conn=conn)
    assert len(log) == 3 and log[0]["label_emosi"] == "sad", "urutan harus terbaru dulu"
    assert log[0]["id_model"] == 1, "otomatis menempel ke model terakhir"

    r = {x["label_emosi"]: x for x in rekap(conn)}
    assert r["happy"]["jumlah"] == 2
    assert abs(r["happy"]["rata_confidence"] - 0.81) < 1e-9
    print("logger OK")

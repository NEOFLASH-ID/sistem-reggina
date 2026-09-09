"""Analisis karakteristik emosional pengunjung — olahan dari tabel `log_deteksi`.

Modul evaluasi menilai *modelnya* (akurasi, F1). Modul ini menjawab pertanyaan
penelitian: emosi apa yang dominan muncul di lokasi, kapan, dan seberapa yakin
sistem saat mendeteksinya.
"""
from . import db
from .dataset_manager import LABELS


def distribusi(conn=None, id_model=None):
    """Jumlah dan persentase tiap emosi, plus rata-rata confidence-nya."""
    conn = conn or db.connect()
    q = ("SELECT label_emosi, COUNT(*) AS jumlah, AVG(confidence_score) AS rata_confidence"
         " FROM log_deteksi")
    arg = ()
    if id_model is not None:
        q += " WHERE id_model=?"
        arg = (id_model,)
    baris = [dict(r) for r in conn.execute(q + " GROUP BY label_emosi", arg)]
    total = sum(b["jumlah"] for b in baris) or 1
    for b in baris:
        b["persentase"] = b["jumlah"] / total * 100
    urut = {l: i for i, l in enumerate(LABELS)}
    return sorted(baris, key=lambda b: urut.get(b["label_emosi"], 99))


def dominan(conn=None, id_model=None):
    """Emosi paling sering terdeteksi -> (label, jumlah, persentase)."""
    d = distribusi(conn, id_model)
    if not d:
        return None
    b = max(d, key=lambda x: x["jumlah"])
    return b["label_emosi"], b["jumlah"], b["persentase"]


def per_jam(conn=None):
    """Sebaran deteksi per jam — untuk melihat pola emosi sepanjang waktu operasional."""
    conn = conn or db.connect()
    return [dict(r) for r in conn.execute(
        "SELECT substr(waktu_deteksi,12,2) AS jam, label_emosi, COUNT(*) AS jumlah"
        " FROM log_deteksi GROUP BY jam, label_emosi ORDER BY jam")]


def naratif(conn=None, id_model=None):
    """Ringkasan kalimat siap pakai untuk pembahasan BAB V."""
    d = distribusi(conn, id_model)
    if not d:
        return "Belum ada data deteksi yang tercatat."
    total = sum(b["jumlah"] for b in d)
    dom = dominan(conn, id_model)
    rinci = ", ".join(f"{b['label_emosi']} {b['persentase']:.1f}%" for b in d)
    yakin = max(d, key=lambda b: b["rata_confidence"])
    ragu = min(d, key=lambda b: b["rata_confidence"])
    return (
        f"Dari {total} deteksi yang tercatat, sebaran emosi yang terekam adalah {rinci}. "
        f"Emosi dominan adalah {dom[0]} ({dom[2]:.1f}% dari seluruh deteksi). "
        f"Sistem paling yakin ketika mengenali {yakin['label_emosi']} "
        f"(rata-rata confidence {yakin['rata_confidence']:.2f}) dan paling ragu pada "
        f"{ragu['label_emosi']} (rata-rata {ragu['rata_confidence']:.2f}), "
        "sejalan dengan hasil evaluasi model pada confusion matrix.")


if __name__ == "__main__":
    conn = db.connect(":memory:")
    contoh = [("2026-09-03 09:15:00", "happy", 0.90), ("2026-09-03 09:40:00", "happy", 0.80),
              ("2026-09-03 10:05:00", "sad", 0.50), ("2026-09-03 10:30:00", "neutral", 0.60)]
    for w, l, c in contoh:
        conn.execute("INSERT INTO log_deteksi (waktu_deteksi,label_emosi,confidence_score)"
                     " VALUES (?,?,?)", (w, l, c))
    conn.commit()

    d = {b["label_emosi"]: b for b in distribusi(conn)}
    assert d["happy"]["jumlah"] == 2 and abs(d["happy"]["persentase"] - 50) < 1e-9
    assert abs(d["happy"]["rata_confidence"] - 0.85) < 1e-9
    assert dominan(conn)[0] == "happy"

    jam = {(r["jam"], r["label_emosi"]): r["jumlah"] for r in per_jam(conn)}
    assert jam[("09", "happy")] == 2 and jam[("10", "sad")] == 1, jam

    t = naratif(conn)
    assert "dominan adalah happy" in t and "50.0%" in t
    print("analysis OK\n" + t)

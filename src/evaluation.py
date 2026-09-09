"""UC8: Evaluasi model — accuracy, precision, recall, F1, confusion matrix (bahan BAB V)."""
from pathlib import Path

import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)

from .dataset_manager import LABELS, ROOT


def evaluasi(model, X_test, y_test, labels=LABELS):
    """Kembalikan dict berisi akurasi, metrik per kelas, rata-rata, dan confusion matrix."""
    labels = [l for l in labels if l in set(y_test)]
    y_pred = model.predict(X_test)
    lap = classification_report(y_test, y_pred, labels=labels,
                                output_dict=True, zero_division=0)
    return {
        "labels": labels,
        "akurasi": lap["accuracy"],
        "per_kelas": {l: lap[l] for l in labels},
        "rata_rata": lap["macro avg"],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
    }


def simpan_confusion_matrix(hasil, ke=ROOT / "data" / "confusion_matrix.png"):
    """Tulis gambar confusion matrix untuk dilampirkan di BAB V."""
    import matplotlib
    matplotlib.use("Agg")               # tanpa GUI: aman di dalam container
    import matplotlib.pyplot as plt

    d = ConfusionMatrixDisplay(hasil["confusion_matrix"], display_labels=hasil["labels"])
    d.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - HOG + SVM")
    plt.tight_layout()
    Path(ke).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ke, dpi=150)
    plt.close()
    return ke


def teks(hasil):
    """Tabel metrik siap salin ke laporan."""
    baris = [f"Akurasi keseluruhan: {hasil['akurasi']:.4f}", "",
             f"{'kelas':<10}{'precision':>10}{'recall':>10}{'f1-score':>10}{'jumlah':>8}"]
    for l, m in hasil["per_kelas"].items():
        baris.append(f"{l:<10}{m['precision']:>10.4f}{m['recall']:>10.4f}"
                     f"{m['f1-score']:>10.4f}{int(m['support']):>8}")
    r = hasil["rata_rata"]
    baris.append(f"{'rata-rata':<10}{r['precision']:>10.4f}{r['recall']:>10.4f}"
                 f"{r['f1-score']:>10.4f}{int(r['support']):>8}")
    return "\n".join(baris)


if __name__ == "__main__":
    class Tebakan:
        """Model palsu: benar untuk semua kecuali satu, biar angka metriknya bisa diperiksa."""
        def predict(self, X):
            p = np.array(["happy", "sad", "angry", "neutral"] * 2)
            p[0] = "sad"
            return p

    y = np.array(["happy", "sad", "angry", "neutral"] * 2)
    h = evaluasi(Tebakan(), np.zeros((8, 3)), y)
    assert abs(h["akurasi"] - 7 / 8) < 1e-9, h["akurasi"]
    assert h["confusion_matrix"].shape == (4, 4)
    assert h["confusion_matrix"].sum() == 8
    assert h["per_kelas"]["happy"]["recall"] == 0.5      # 1 dari 2 happy salah tebak
    assert "Akurasi keseluruhan" in teks(h)
    print("evaluation OK\n" + teks(h))

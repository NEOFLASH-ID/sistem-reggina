"""Praproses citra — tahap antara deteksi wajah dan ekstraksi fitur (BAB II, BAB IV 4.3.5).

Dipisah sebagai modul sendiri karena tahap ini harus identik saat pelatihan dan
saat deteksi real-time; kalau berbeda, vektor HOG tidak sebanding dan akurasi jatuh.
"""
import cv2

SIZE = 128          # seluruh citra diseragamkan 128x128 piksel


def praproses(citra, size=SIZE):
    """Citra apa pun -> grayscale, ukuran seragam size x size, siap diekstraksi HOG.

    Dua langkah: (1) konversi ke aras keabuan agar fitur tidak bergantung warna
    kulit/pencahayaan berwarna, (2) penyeragaman ukuran agar panjang vektor HOG
    selalu sama untuk setiap citra.
    """
    if citra.ndim == 3:
        citra = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
    if citra.shape != (size, size):
        citra = cv2.resize(citra, (size, size), interpolation=cv2.INTER_AREA)
    return citra


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)

    warna = rng.integers(0, 256, (300, 200, 3), dtype=np.uint8)
    out = praproses(warna)
    assert out.shape == (SIZE, SIZE) and out.ndim == 2, out.shape

    abu = rng.integers(0, 256, (50, 400), dtype=np.uint8)
    assert praproses(abu).shape == (SIZE, SIZE)

    pas = rng.integers(0, 256, (SIZE, SIZE), dtype=np.uint8)
    assert praproses(pas) is pas, "citra yang sudah sesuai tidak perlu diproses ulang"
    print("preprocessing OK")

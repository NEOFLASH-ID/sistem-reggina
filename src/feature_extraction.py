"""UC3: Ekstraksi Fitur HOG — parameter mengikuti BAB IV 4.3.5."""
import cv2
import numpy as np
from skimage.feature import hog

from .preprocessing import SIZE, praproses

# BAB IV 4.3.5: cell 8x8 px, block 2x2 sel, 9 bin orientasi, normalisasi L2
PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
              block_norm="L2", feature_vector=True)

# 128/8 = 16 sel; block 2x2 geser 1 sel -> 15x15 block; 15*15*2*2*9 = 8100
DIM = (SIZE // 8 - 1) ** 2 * 2 * 2 * 9


def extract(citra):
    """Citra grayscale 128x128 -> vektor fitur HOG (float, panjang DIM)."""
    return hog(praproses(citra), **PARAMS)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (SIZE, SIZE), dtype=np.uint8)

    v = extract(a)
    assert v.shape == (DIM,), (v.shape, DIM)          # dimensi konsisten = syarat input SVM
    assert extract(cv2.resize(a, (300, 200))).shape == (DIM,)   # ukuran lain tetap seragam
    assert extract(cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)).shape == (DIM,)  # BGR ikut jalan

    b = np.zeros((SIZE, SIZE), np.uint8)
    assert not np.allclose(extract(b), v), "citra beda harus beda fitur"
    print(f"feature_extraction OK, dim={DIM}")

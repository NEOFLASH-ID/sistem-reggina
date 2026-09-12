# Panduan Instalasi dan Penggunaan
## Sistem Pengenalan Emosi Wajah (HOG + SVM)

Dokumen ini ditujukan untuk pengguna yang menerima berkas sistem dan ingin
menjalankannya di komputer sendiri. Seluruh pemrosesan berjalan lokal di
komputer Anda: tidak ada data yang dikirim ke internet.

---

## 1. Kebutuhan Komputer

| Kebutuhan | Keterangan |
|---|---|
| Sistem operasi | Windows 10/11, macOS, atau Linux |
| Python | versi 3.9 atau lebih baru |
| RAM | minimal 4 GB (disarankan 8 GB) |
| Ruang penyimpanan | sekitar 2 GB (termasuk pustaka pendukung) |
| Kamera | webcam internal atau USB |
| Koneksi internet | hanya saat instalasi, tidak dibutuhkan saat sistem berjalan |

Tidak memerlukan kartu grafis (GPU) khusus.

---

## 2. Yang Perlu Diinstal

### 2.1 Python

Unduh dari <https://www.python.org/downloads/> lalu pasang.

**Khusus Windows:** saat proses instalasi, centang kotak
**"Add Python to PATH"** di layar pertama. Jika terlewat, perintah `python`
tidak akan dikenali dan instalasi harus diulang.

Pastikan berhasil dengan membuka Terminal (macOS/Linux) atau Command Prompt
(Windows), lalu ketik:

```
python --version
```

Harus muncul tulisan seperti `Python 3.11.9`. Jika di macOS/Linux tidak
dikenali, coba `python3 --version`.

### 2.2 Pustaka pendukung

Tidak perlu dipasang satu per satu. Semuanya tercantum dalam berkas
`requirements.txt` dan dipasang sekaligus pada langkah 3.

Daftar pustaka yang akan terpasang:

| Pustaka | Kegunaan |
|---|---|
| opencv-python | mengakses kamera dan mendeteksi wajah |
| scikit-image | ekstraksi fitur HOG |
| scikit-learn | klasifikasi SVM |
| streamlit | antarmuka aplikasi berbasis browser |
| numpy, pandas, matplotlib, altair | perhitungan dan penyajian grafik |
| joblib | memuat model yang sudah dilatih |

---

## 3. Langkah Instalasi

### 3.1 Unduh sistem dari GitHub

Sistem tersedia di <https://github.com/NEOFLASH-ID/sistem-reggina>.

Cara termudah, lewat Terminal (macOS/Linux) atau Command Prompt (Windows).
Masuk dulu ke folder tempat sistem ingin diletakkan, misalnya `Documents`,
lalu jalankan:

```
git clone https://github.com/NEOFLASH-ID/sistem-reggina.git
cd sistem-reggina
```

Unduhan berukuran sekitar 120 MB karena sudah termasuk model yang terlatih,
sehingga sistem langsung bisa dipakai tanpa proses pelatihan.

Bila perintah `git` belum dikenali, pasang Git lebih dulu dari
<https://git-scm.com/downloads>. Alternatif tanpa Git: buka halaman GitHub di
atas, klik tombol hijau **Code**, pilih **Download ZIP**, lalu ekstrak.

**Memperbarui ke versi terbaru.** Bila kelak ada perbaikan, cukup jalankan
perintah berikut di dalam folder sistem, tanpa perlu memasang ulang apa pun:

```
git pull
```

Dataset mentah tidak disertakan karena berukuran besar dan terikat lisensi
penyedianya. Dataset hanya dibutuhkan bila ingin melatih ulang model, bukan
untuk menjalankan sistem.

**Mengambil dataset (opsional).** Bila ingin melatih ulang model atau melihat
berkas citra latihnya, jalankan satu perintah berikut setelah langkah 4
selesai. Unduhan sekitar 258 MB dan memakan waktu 2-3 menit:

**Windows:** `.venv\Scripts\python siapkan_dataset.py`
**macOS / Linux:** `.venv/bin/python siapkan_dataset.py`

Perintah ini mengunduh dataset AffectNet dari Kaggle (tanpa perlu akun),
menyalin 500 citra per emosi, melakukan praproses, lalu mencatatnya ke
database. Aman dijalankan berulang kali: citra yang sudah tercatat tidak
digandakan.

### 3.2 Buka Terminal di folder sistem

- **Windows:** buka folder di File Explorer, ketik `cmd` pada kolom alamat, tekan Enter.
- **macOS:** klik kanan folder, pilih New Terminal at Folder.
- **Linux:** klik kanan di dalam folder, pilih Open in Terminal.

### 3.3 Siapkan lingkungan Python

**Windows:**
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**macOS / Linux:**
```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Proses ini mengunduh sekitar 300 MB dan memakan waktu 3-10 menit tergantung
kecepatan internet. Cukup dilakukan satu kali.

### 3.4 Periksa hasil instalasi

**Windows:**
```
.venv\Scripts\python src\db.py
.venv\Scripts\python -m src.face_detection
```

**macOS / Linux:**
```
.venv/bin/python src/db.py
.venv/bin/python -m src.face_detection
```

Jika muncul `db.py OK` dan `face_detection OK`, instalasi berhasil.

---

## 4. Menjalankan Sistem

**Windows:**
```
.venv\Scripts\streamlit run app\main.py
```

**macOS / Linux:**
```
.venv/bin/streamlit run app/main.py
```

Browser akan terbuka sendiri di alamat <http://localhost:8501>. Bila tidak,
buka alamat tersebut secara manual.

Untuk menghentikan sistem, kembali ke jendela Terminal lalu tekan
**Ctrl + C**.

### Izin kamera

Saat pertama kali digunakan, sistem operasi akan meminta izin akses kamera.
Pilih **Izinkan / Allow**.

Khusus macOS, izin melekat pada aplikasi Terminal yang Anda pakai. Jika
kamera tetap tidak menyala: buka **System Settings > Privacy & Security >
Camera**, aktifkan untuk aplikasi Terminal tersebut, lalu tutup Terminal
sepenuhnya (Cmd + Q) dan buka kembali.

---

## 5. Cara Menggunakan

Sistem memiliki dua halaman, dipilih melalui menu di sisi kiri.

### 5.1 Halaman "Deteksi Real-Time"

Halaman utama untuk mengenali emosi. Tersedia dua mode:

**Mode Jepret** (disarankan)
1. Pilih **Jepret**.
2. Klik tombol kamera, izinkan akses bila diminta.
3. Arahkan wajah ke kamera, klik **Take Photo**.
4. Sistem menampilkan kotak penanda wajah, nama emosi, dan tingkat keyakinan
   dalam persen.

Mode ini menggunakan kamera perangkat yang sedang Anda pakai, sehingga tetap
berfungsi bila sistem diakses dari komputer atau ponsel lain.

**Mode Live**
1. Pilih **Live**, lalu aktifkan tombol **Nyalakan kamera**.
2. Video berjalan terus dan label emosi diperbarui mengikuti ekspresi.
3. Matikan tombol tersebut untuk berhenti.

Mode ini membaca kamera komputer tempat sistem dijalankan, bukan kamera
penonton. Bila sistem dibuka dari perangkat lain, gunakan mode Jepret.

**Empat emosi yang dikenali:** Senang (happy), Sedih (sad), Marah (angry),
dan Netral (neutral). Sistem membaca satu wajah dalam satu waktu, yaitu wajah
yang paling besar terlihat di layar.

Setiap hasil deteksi tersimpan otomatis ke basis data, dan hanya dicatat saat
emosi berubah sehingga catatan tidak membengkak.

### 5.2 Halaman "Admin"

Berisi empat tab.

**Tab Dataset** — menampilkan jumlah citra latih per emosi. Tombol
*Ingest data/raw* dipakai bila ada citra baru yang ditambahkan ke folder
`data/raw/<nama_emosi>/`. Menekan tombol ini berulang kali aman: citra yang
sudah tercatat tidak akan digandakan.

**Tab Training & Evaluasi** — menampilkan:
- Riwayat pelatihan model beserta akurasinya
- Akurasi, precision, recall, dan F1-score untuk tiap emosi
- Confusion matrix, yaitu tabel yang memperlihatkan emosi mana yang sering
  tertukar (arahkan kursor ke setiap kotak untuk melihat angkanya)

Tombol *Latih model sekarang* hanya diperlukan bila dataset berubah. Proses
ini memakan waktu sekitar 21 menit dan halaman tidak boleh ditutup selama
berlangsung. Untuk pemakaian biasa, tombol ini tidak perlu disentuh karena
model sudah disertakan.

**Tab Analisis Emosi** — ringkasan karakteristik emosional dari seluruh
deteksi yang pernah terjadi: emosi yang paling dominan, persentase tiap
emosi, rata-rata tingkat keyakinan, dan sebaran deteksi per jam.

**Tab Log Deteksi** — daftar rinci setiap deteksi beserta waktu, emosi, dan
tingkat keyakinannya.

## 6. Bila Terjadi Masalah

| Gejala | Penyebab dan penanganan |
|---|---|
| `python` tidak dikenali (Windows) | Python belum ditambahkan ke PATH. Pasang ulang Python dan centang "Add Python to PATH". |
| Kamera tidak menyala | Izin kamera belum diberikan, atau kamera sedang dipakai aplikasi lain (Zoom, Google Meet, Camera). Tutup aplikasi tersebut, sebab kamera hanya bisa dipakai satu aplikasi dalam satu waktu. |
| Muncul pesan "Model belum ada" | Berkas `models/svm_model.joblib` tidak ikut tersalin. Salin ulang berkas tersebut ke folder `models/`. |
| Metrik evaluasi tidak muncul | Berkas `data/split_uji.npz` tidak ikut tersalin. |
| Wajah tidak terdeteksi | Dekatkan wajah ke kamera, pastikan wajah menghadap depan dan ruangan cukup terang. Sistem tidak mengenali wajah yang terlalu miring atau membelakangi cahaya. |
| Alamat localhost:8501 tidak terbuka | Port sedang dipakai program lain. Jalankan dengan menambahkan `--server.port 8502`, lalu buka <http://localhost:8502>. |

---

## 7. Catatan Penggunaan

- Seluruh pemrosesan berjalan di komputer Anda. Tidak ada citra maupun hasil
  deteksi yang dikirim ke server luar.
- Sistem mengenali empat emosi dan satu wajah dalam satu waktu.
- Emosi Senang paling akurat dikenali. Emosi Sedih, Marah, dan Netral lebih
  sering tertukar satu sama lain karena bentuk mulutnya serupa; hal ini
  terlihat pada confusion matrix di halaman Admin.
- Hasil deteksi disertai tingkat keyakinan. Keyakinan di bawah 50 persen
  sebaiknya dianggap kurang meyakinkan.

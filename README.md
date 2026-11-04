# Document Image Classification (RVL-CDIP) dengan PyTorch

Proyek ini membangun model *deep learning* untuk mengklasifikasikan halaman dokumen hasil scan ke dalam **16 kategori dokumen bisnis**, menggunakan subset dataset **RVL-CDIP** (32.003 gambar train dan 7.994 gambar test). Hasil prediksi disimpan sebagai `submission.csv` dengan format yang sama seperti `sample_submission.csv`.

Metrik evaluasi: **Categorical Accuracy** (persentase prediksi yang sama dengan label sebenarnya).

---

## Daftar Isi

1. [Dataset dan Kelas](#dataset-dan-kelas)
2. [Struktur Proyek](#struktur-proyek)
3. [Penjelasan Pipeline](#penjelasan-pipeline)
4. [Arsitektur Model](#arsitektur-model)
5. [Konfigurasi dan Hyperparameter](#konfigurasi-dan-hyperparameter)
6. [Format Output](#format-output)
7. [Cara Menjalankan Program](#cara-menjalankan-program)
8. [Troubleshooting](#troubleshooting)

---

## Dataset dan Kelas

Setiap dokumen termasuk tepat satu dari 16 kelas berikut:

| Label | Kategori | Label | Kategori |
|------:|----------|------:|----------|
| 0 | advertisement | 8 | memo |
| 1 | budget | 9 | news_article |
| 2 | email | 10 | presentation |
| 3 | file_folder | 11 | questionnaire |
| 4 | form | 12 | resume |
| 5 | handwritten | 13 | scientific_publication |
| 6 | invoice | 14 | scientific_report |
| 7 | letter | 15 | specification |

Distribusi data train relatif seimbang (sekitar 1.950–2.060 gambar per kelas).

Tantangan khas data ini: tata letak yang beragam, noise hasil scan (kemiringan, kontras rendah, artefak kompresi fax), serta campuran teks cetak, tulisan tangan, dan stempel. Beberapa file TIFF juga diketahui rusak atau terpotong.

File yang dibutuhkan:

| File | Isi |
|------|-----|
| `train.csv` | kolom `filename` (path gambar) dan `label` (0–15) |
| `test.csv` | kolom `id` (misalnya `test_00001`) dan `filename` |
| `sample_submission.csv` | contoh format file jawaban (`id,label`) |
| `train/` dan `test/` | gambar dokumen berformat `.tif` |

---

## Struktur Proyek

```
document-image-classification/
├── .venv/                    # virtual environment (tidak di-commit)
├── notebooks/                # notebook eksperimen
├── submission/
│   └── submission.csv        # hasil prediksi (dibuat oleh script)
├── train/
│   └── train/<nama_kelas>/*.tif
├── test/
│   └── test/test_xxxxx.tif
├── train.csv
├── test.csv
├── sample_submission.csv
└── README.md
```

Script mendukung dua bentuk folder gambar, yaitu bersarang (`train/train/...`, seperti hasil ekstrak dari Kaggle) maupun tidak bersarang (`train/...`). Penanganannya ada di fungsi `resolve()`.

---

## Penjelasan Pipeline

Seluruh proses ada di satu file, `scripts/train_and_submit.py`, dan berjalan berurutan sebagai berikut.

### 1. Persiapan lingkungan
- **Pembersihan memori otomatis.** Di bagian awal, script menghapus sisa variabel (`model`, `opt`, dan lainnya) dan traceback dari run sebelumnya. Ini mencegah memori GPU penuh ketika sel dijalankan ulang di Jupyter.
- **Pemilihan perangkat otomatis.** Urutan prioritas: `cuda` (GPU NVIDIA), lalu `mps` (GPU Apple Silicon), lalu `cpu`. Mixed precision (AMP) hanya diaktifkan di CUDA.
- **Argumen konfigurasi.** Menggunakan `argparse` dengan `parse_known_args()`, sehingga aman dijalankan di terminal maupun di Jupyter Notebook (yang menyisipkan argumen tambahan miliknya sendiri).

### 2. Pemuatan data
- `train.csv` dan `test.csv` dibaca dengan pandas.
- Fungsi `resolve(filename)` mengubah path di CSV menjadi path file yang benar di disk, dan dicek pada satu sampel sebelum training dimulai (*fail fast*).
- **Pembagian validasi:** 5% dari data train diambil secara acak (`random_state=0`) sebagai data validasi, 95% sisanya untuk training.

### 3. Dataset dan transformasi gambar
Kelas `DS` (turunan `torch.utils.data.Dataset`) membaca gambar, mengubahnya ke RGB, lalu menerapkan transformasi.

| Tahap | Train | Validasi / Test |
|-------|:-----:|:---------------:|
| Resize ke `size × size` | ya | ya |
| RandomAffine (rotasi ±3°, geser 2%, skala 0,95–1,05, latar putih) | ya | tidak |
| ToTensor | ya | ya |
| Normalisasi ImageNet (mean/std) | ya | ya |

Pilihan desain augmentasi:
- **Tanpa flip horizontal**, karena tata letak dokumen punya arah baca yang bermakna (membalik gambar bisa membuat dokumen tidak realistis).
- Rotasi dan skala dibuat kecil untuk meniru kemiringan dan variasi hasil scan tanpa merusak struktur dokumen.
- Area kosong akibat transformasi diisi putih (`fill=255`), sesuai warna latar kertas.

**Penanganan file rusak.** Jika sebuah gambar gagal dibaca, script tidak berhenti. Gambar diganti halaman putih dan dicetak satu peringatan `[WARN] gagal baca <file>`. Pengaturan `LOAD_TRUNCATED_IMAGES = True` juga mencoba membaca TIFF yang terpotong. Ini berlaku juga untuk data test, sehingga jumlah baris submission selalu lengkap.

### 4. Training
- **Loss:** `CrossEntropyLoss` dengan *label smoothing* 0,1, untuk mengurangi *overconfidence* dan membantu kelas yang mirip (misalnya `letter` vs `memo`, atau `scientific_publication` vs `scientific_report`).
- **Optimizer:** `AdamW` dengan learning rate 1e-4 dan weight decay 0,05.
- **Scheduler:** `OneCycleLR` (learning rate naik lalu turun selama training).
- **Mixed precision:** `GradScaler` dan `autocast` aktif hanya pada CUDA.
- **MPS:** cache memori dikosongkan setiap langkah (`torch.mps.empty_cache()`) dan `num_workers=0` karena DataLoader multi-proses sering bermasalah di MPS.
- Setelah setiap epoch, script mengevaluasi data validasi dan mencetak `val_acc`.

### 5. Prediksi dan pembuatan submission
- Model dalam mode `eval()` memprediksi seluruh data test (`argmax` dari logit).
- Prediksi dicocokkan berdasarkan kolom `id` ke `sample_submission.csv` (disalin, lalu kolom `label` diisi) dan disimpan dengan `index=False`.
- Ada pengecekan otomatis (`assert`): jumlah baris sama dengan sample dan semua label berada pada rentang 0–15.
- Folder tujuan dibuat otomatis jika belum ada.

---

## Arsitektur Model

- **Model dasar:** ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`) dari pustaka [`timm`](https://github.com/huggingface/pytorch-image-models), berupa CNN modern yang sudah dilatih (*pre-trained*) pada ImageNet-22k lalu di-fine-tune pada ImageNet-1k.
- **Transfer learning:** lapisan klasifikasi akhir diganti menjadi 16 kelas (`num_classes=16`), lalu seluruh jaringan di-*fine-tune* pada data dokumen.
- Arsitektur dapat diganti lewat argumen `--model`, misalnya `resnet50`, `efficientnet_b0`, atau model Transformer seperti `vit_base_patch16_224` (gunakan `--size 224`). Semua pilihan tetap berbasis neural network dan PyTorch.

---

## Konfigurasi dan Hyperparameter

| Argumen | Default | Keterangan |
|---------|---------|------------|
| `--data_dir` | `DATA_DIR` di script | folder berisi CSV, `train/`, dan `test/` |
| `--out` | `OUT_PATH` di script | lokasi file submission |
| `--epochs` | 3 | jumlah epoch |
| `--bs` | 8 | batch size (kecil agar muat di memori GPU Mac) |
| `--size` | 224 | resolusi gambar (piksel, persegi) |
| `--model` | `convnext_tiny.fb_in22k_ft_in1k` | nama model dari `timm` |

Nilai default dipilih agar muat di Mac dengan GPU Apple Silicon (MPS). Untuk GPU yang lebih besar (Colab/Kaggle/CUDA), disarankan menaikkan ke `--size 384 --bs 32 --epochs 5` agar akurasi lebih baik, terutama untuk kelas yang bergantung pada detail teks kecil.

Dua variabel di bagian atas script juga bisa diedit langsung tanpa argumen terminal:

```python
DATA_DIR = "."                          # folder berisi train.csv, test.csv, folder train/ & test/
OUT_PATH = "submission/submission.csv"  # tempat hasil disimpan
```

---

## Format Output

File `submission.csv` memiliki header dan 7.994 baris, sama persis dengan `sample_submission.csv`:

```
id,label
test_00001,7
test_00002,12
test_00003,5
...
```

- `id`: ID gambar test.
- `label`: bilangan bulat 0–15 sesuai tabel kelas.

---

## Cara Menjalankan Program

### Prasyarat
- Python 3.9 atau lebih baru
- Git (opsional, untuk meng-clone repositori)
- GPU sangat disarankan (NVIDIA CUDA, Apple Silicon MPS, atau GPU gratis di Google Colab/Kaggle). CPU bisa dipakai tetapi sangat lambat.

### 1. Siapkan folder dan data
Letakkan `train.csv`, `test.csv`, `sample_submission.csv`, folder `train/`, dan folder `test/` di folder proyek, sesuai [struktur proyek](#struktur-proyek). Simpan `train_and_submit.py` di folder `scripts/`.

### 2. Membuat dan mengaktifkan virtual environment

Buka terminal di folder proyek (folder yang berisi `train.csv`).

**macOS / Linux**
```bash
# membuat environment (cukup sekali)
python3 -m venv .venv

# mengaktifkan environment (setiap membuka terminal baru)
source .venv/bin/activate
```

**Windows (Command Prompt)**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Environment aktif jika di awal baris terminal muncul `(.venv)`. Untuk menonaktifkan, ketik `deactivate`.

Jika memakai VS Code, pilih interpreter `.venv` lewat *Command Palette* → **Python: Select Interpreter**. Untuk notebook, pilih kernel dari `.venv` di pojok kanan atas.

### 3. Menginstal dependensi

Dengan environment aktif:

```bash
pip install --upgrade pip
pip install torch torchvision timm pandas numpy pillow
```

Opsional, untuk menghilangkan peringatan progress bar di Jupyter:

```bash
pip install ipywidgets
```

Cek instalasi dan perangkat GPU:

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available(), 'mps:', torch.backends.mps.is_available())"
```

Catatan: model pre-trained diunduh otomatis dari Hugging Face Hub saat pertama kali dijalankan, jadi koneksi internet diperlukan. Peringatan soal `HF_TOKEN` boleh diabaikan.

### 4. Menjalankan lewat terminal

```bash
python scripts/train_and_submit.py --data_dir . --out submission/submission.csv
```

Contoh dengan pengaturan sendiri:

```bash
# Mac / GPU kecil (hemat memori)
python scripts/train_and_submit.py --epochs 3 --size 224 --bs 8

# GPU besar (Colab/Kaggle/CUDA)
python scripts/train_and_submit.py --epochs 5 --size 384 --bs 32

# Mengganti model
python scripts/train_and_submit.py --model efficientnet_b0
```

### 5. Menjalankan lewat Jupyter Notebook
1. Salin isi `train_and_submit.py` ke satu sel notebook.
2. Edit `DATA_DIR` dan `OUT_PATH` di bagian atas (jika notebook berada di folder `notebooks/`, gunakan `DATA_DIR = ".."` dan `OUT_PATH = "../submission/submission.csv"`, atau path lengkap).
3. Ubah nilai `default=` pada `p.add_argument(...)` bila ingin mengubah `epochs`, `bs`, atau `size`.
4. Jalankan sel. Jika perlu mengulang setelah error, lakukan **Restart Kernel** terlebih dahulu agar memori GPU benar-benar bebas.

### 6. Output yang diharapkan

```
device: mps
CONFIG -> size=224 bs=8 epochs=3 model=convnext_tiny.fb_in22k_ft_in1k
epoch 1/3  val_acc=0.xxxx
epoch 2/3  val_acc=0.xxxx
epoch 3/3  val_acc=0.xxxx
Saved submission/submission.csv (7994, 2)
```

Sel tidak menampilkan output sampai satu epoch selesai, jadi wajar jika terlihat diam selama beberapa waktu. Di GPU Apple Silicon, satu epoch bisa memakan waktu beberapa jam.

---

## Troubleshooting

| Masalah | Penyebab dan solusi |
|---------|---------------------|
| `unrecognized arguments: --f=...json` | Script dijalankan di Jupyter dengan `parse_args()`. Gunakan versi terbaru yang memakai `parse_known_args()`. |
| `FileNotFoundError: ... train.csv` | `DATA_DIR` salah. Pastikan menunjuk ke folder yang berisi `train.csv`, dan periksa huruf besar/kecil nama folder. |
| `MPS backend out of memory` | Restart kernel, turunkan `--bs` (misalnya 4) atau `--size`, atau ganti ke model lebih kecil. Jangan mengaktifkan `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` karena dapat membuat sistem hang. |
| `UnidentifiedImageError` | File TIFF rusak. Versi terbaru menanganinya otomatis (gambar diganti halaman putih dan muncul `[WARN]`). |
| Training sangat lambat | Gunakan GPU (Colab/Kaggle), atau uji coba dulu dengan sebagian data sebelum menjalankan penuh. |

---

## Catatan

- Script bersifat *self-contained* (satu file) agar mudah direproduksi. Pembagian validasi memakai `random_state=0`, tetapi hasil training tetap dapat sedikit berbeda antar-run karena sifat acak pada augmentasi dan inisialisasi.
- Folder `.venv/`, gambar dataset, dan file besar sebaiknya dimasukkan ke `.gitignore` agar tidak ikut ter-commit:

```
.venv/
train/
test/
*.tif
```
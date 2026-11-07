# Credit Card Fraud Detection using Deep Neural Network

Proyek Mata Kuliah Pembelajaran Mendalam (Semester Ganjil TA 2026/2027).

Model DNN fully-connected untuk mengklasifikasikan transaksi kartu kredit sebagai **normal** atau
**fraud** pada dataset yang sangat tidak seimbang (fraud hanya 0,173%). Dua fitur unik:

1. **Optimasi threshold keputusan** — threshold dipilih dari data *validation* dengan
   memaksimalkan F1-score, lalu dipakai apa adanya pada data *testing* dan aplikasi.
2. **Kompresi model ke TensorFlow Lite** (dynamic-range int8): 10 KB, keputusan sama dengan
   model Keras pada 42.558 dari 42.559 transaksi uji, dipakai oleh aplikasi Streamlit.

Tiga strategi fungsi loss dibandingkan (BCE, BCE berbobot kelas, Focal Loss; 3 seed) dan
model final dipilih otomatis dari data validation (`select_final.py`).

## Hasil model final (data testing, threshold 0,1276)

| Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| 0,9375 | 0,6338 | 0,7563 | 0,8073 | 0,9724 |

## Dataset

[Credit Card Fraud Detection — ULB Machine Learning Group (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284.807 transaksi, 31 kolom (`Time`, `V1`–`V28` hasil PCA, `Amount`, `Class`).

Unduh `creditcard.csv` dari Kaggle dan simpan sebagai `data/creditcard.csv`
(file ~144 MB, tidak disertakan di repositori karena melebihi batas ukuran GitHub).

## Pemetaan langkah RTM ke implementasi

| Langkah RTM | Implementasi |
|---|---|
| 3. Eksplorasi & praproses (pembersihan, normalisasi, split latih/validasi/uji) | `src/eda.py`, `src/preprocessing.py` |
| 4. Input pipeline `tf.data` + arsitektur MLP | `src/data_pipeline.py`, `src/model.py` |
| 5. Loss function, dropout, batch normalization, Adam/RMSProp, LR scheduling, early stopping | `src/model.py`, `src/train.py` |
| 6. Evaluasi (accuracy, precision, recall, F1, confusion matrix) + iterasi perbaikan | `src/evaluate.py`, `src/select_final.py`, `src/plots.py` |
| 7. Dokumentasi arsitektur, hyperparameter, hasil eksperimen | Laporan PDF, `notebooks/`, `outputs/results/` |
| Luaran (c) notebook + instruksi + anggota | `notebooks/Proyek_Deteksi_Fraud_DNN.ipynb`, README ini |
| Luaran (d) source code lengkap | `src/`, `app/` |

## Anggota kelompok

Kelompok **Blueprints** — Kelas **IF B Sore**

| NIM | Nama | Peran |
|---|---|---|
| 231112189 | Nanda Fachriza Artin | Ketua & ML Engineer |
| 231110155 | Arman Maulana | Data Analyst |
| 231110255 | Nyoto Prasetyo | Data Engineer |
| 231112549 | Mhd. Abi Manyu | Evaluator |
| 231110656 | Celvin | App Developer & Deployment |

## Struktur repositori

```
├── app/app.py              # Aplikasi Streamlit (inferensi, model TFLite)
├── notebooks/              # notebook Colab yang menjalankan seluruh pipeline
├── .streamlit/config.toml  # konfigurasi tampilan aplikasi
├── data/creditcard.csv     # dataset (unduh manual, tidak ikut di repo)
├── data/contoh_transaksi_uji.csv  # 20 transaksi data uji untuk demo aplikasi
├── outputs/
│   ├── figures/            # grafik EDA & evaluasi
│   ├── models/             # model .keras, scaler, threshold
│   └── results/            # metrik (JSON), riwayat training, log
├── src/
│   ├── config.py           # konfigurasi & hyperparameter
│   ├── preprocessing.py    # pembersihan, rekayasa fitur, split, scaling
│   ├── data_pipeline.py    # pipeline input tf.data
│   ├── eda.py              # exploratory data analysis
│   ├── model.py            # arsitektur DNN + Binary Focal Loss
│   ├── evaluate.py         # metrik + optimasi threshold (fitur unik #1)
│   ├── train.py            # pelatihan semua skenario
│   ├── select_final.py     # pemilihan model final dari data validation
│   ├── export_tflite.py    # kompresi TFLite (fitur unik #2)
│   ├── error_analysis.py   # analisis TP/FN/FP model final
│   ├── make_sample.py      # membuat contoh data uji untuk demo
│   └── plots.py            # grafik hasil evaluasi
└── requirements.txt
```

## Cara menjalankan

**Cara termudah (Google Colab):**

1. Buka [colab.research.google.com](https://colab.research.google.com) → *File → Upload notebook*
   → pilih `notebooks/Proyek_Deteksi_Fraud_DNN.ipynb`.
2. *Runtime → Run all*. Notebook mengambil kode dengan salah satu cara: clone dari GitHub
   (isi `REPO_URL` di sel pertama), file `Source_Code_Credit_Card_Fraud_DNN.zip` di Google Drive,
   atau unggah zip tersebut saat diminta.
3. Dataset dibaca dari `data/creditcard.csv`; bila belum ada, notebook mencarinya di Google
   Drive, mencoba mengunduh dari Kaggle (`kagglehub`), atau meminta Anda mengunggah
   `creditcard.csv`.
4. Tidak perlu `pip install`: TensorFlow, scikit-learn, pandas, dan matplotlib sudah ada di Colab.
   Pelatihan 14 run ±20–25 menit di CPU. Set `LATIH_ULANG = False` untuk memakai model yang
   sudah dilatih (±1 menit).

**Lokal (terminal):**

```bash
pip install -r requirements.txt
cd src
python eda.py      # grafik & ringkasan EDA
python train.py          # melatih 14 run (8 skenario, beberapa seed)
python select_final.py   # pilih model final + threshold
python export_tflite.py  # kompresi TFLite
python error_analysis.py
python plots.py          # grafik evaluasi
cd ..
streamlit run app/app.py
```

## Menjalankan aplikasi

Aplikasi Streamlit dijalankan secara lokal dengan `streamlit run app/app.py`, lalu dibuka di
browser (`http://localhost:8501`). Untuk mencoba, centang **Gunakan contoh data uji** pada tab
*Unggah CSV*, atau pilih salah satu contoh pada tab *Input manual*. Repositori ini juga dapat
di-deploy langsung ke Streamlit Community Cloud dengan main file path `app/app.py`.

Semua angka di laporan berasal dari `outputs/results/all_results.json` (seed 42 untuk model final).

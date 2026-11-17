# KecerdasanBuatan-Kelompok1

Proyek Tugas Besar Kecerdasan Buatan 2026/2027 untuk dataset Predictive Maintenance (AI4I 2020). Fokus utama pada pemahaman data, preprocessing, baseline neural network, serta kerangka awal dashboard Streamlit sesuai milestone Minggu 12.

## Tujuan Proyek
- Menganalisis dataset predictive maintenance dengan target `Machine failure`.
- Melakukan EDA dan validasi data untuk memastikan target, tipe problem, dan split data aman.
- Membangun baseline model neural network sebagai acuan eksperimen.
- Menyiapkan repository dan framework Streamlit awal yang siap dikembangkan pada minggu berikutnya untuk GA, PSO, fuzzy logic, dan dashboard hybrid.

## Anggota Kelompok 1, Topik : Predictive Maintenance AI4I 2020
- Hasbi Ma'arif (04241034)
- Erika Tri Wulandari ()

## Dataset
Sumber: UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset

Kolom utama:
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`

## Struktur Repository
```text
KecerdasanBuatan-Kelompok1/
├── app/
│   └── streamlit.py
├── data/
│   ├── processed/
│   │   └── processed.csv
│   └── raw/
│       └── ai4i2020.csv
├── src/
│   ├── EDA.py
│   ├── inspect_data.py
│   ├── model.py
│   ├── nn_mlp_ai4i2020.py
│   ├── preprocessing.py
│   └── evaluation.py
├── README.md
├── requirements.txt
└── .gitignore

## Langkah Cepat
1. Instal dependensi:
``pip install -r requirements.txt``

2. Jalankan EDA:
``python src/EDA.py``

3. Jalankan baseline model:
``python src/nn_mlp_ai4i2020.py``

4. Jalankan dashboard Streamlit:
``streamlit run app/streamlit.py``

## Reproducibility
- Gunakan versi library yang stabil seperti yang terdaftar di `requirements.txt`.
- Simpan preprocessing pipeline dan model hasil training untuk kebutuhan evaluasi lanjutan.
- Dokumentasikan semua variabel lingkungan dan command yang digunakan agar eksperimen dapat diulang.

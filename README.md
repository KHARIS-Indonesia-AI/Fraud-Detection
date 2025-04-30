# Sistem Deteksi Fraud dengan Machine Learning

Sebuah sistem cerdas untuk mendeteksi transaksi mencurigakan pada data perbankan dan kartu kredit menggunakan teknik machine learning.

## Latar Belakang

Kasus penipuan (fraud) dalam transaksi perbankan dan kartu kredit terus meningkat dan menimbulkan kerugian finansial yang signifikan bagi lembaga keuangan maupun nasabah. Seiring berkembangnya teknologi, metode penipuan juga semakin canggih dan sulit dideteksi dengan pendekatan konvensional.

Machine learning menawarkan pendekatan yang lebih efektif karena mampu:
- Menganalisis jutaan transaksi dan menemukan pola tersembunyi
- Mengenali perilaku mencurigakan yang mungkin terlewatkan oleh sistem tradisional
- Menyesuaikan diri dengan tren fraud terbaru secara lebih cepat
- Mengurangi kasus "false alarm" yang sering mengganggu nasabah

Proyek ini dikembangkan untuk membantu lembaga keuangan mengidentifikasi transaksi fraud secara akurat dan real-time, dengan meminimalkan gangguan pada transaksi normal nasabah.

## Dataset

Data yang digunakan adalah `fraudTrain.csv` yang berisi catatan transaksi dengan berbagai informasi seperti:
- Jumlah transaksi (`amt`)
- Kategori pembelian (`category`)
- Lokasi pengguna dan merchant (`lat`, `long`, `merch_lat`, `merch_long`)
- Waktu transaksi (`trans_date_trans_time`)
- Informasi demografis pengguna
- Label transaksi (`is_fraud`) yang menunjukkan apakah transaksi tersebut fraud atau tidak

## Tahapan Analisis

Kode dibagi menjadi 12 tahapan utama:

1. **Loading Data**
   - Membaca dataset dari Google Drive
   - Melihat informasi dasar dataset

2. **Exploratory Data Analysis (EDA)**
   - Analisis struktur dan distribusi data
   - Identifikasi missing values
   - Visualisasi proporsi fraud vs non-fraud

3. **Data Preprocessing**
   - Pembersihan data dan penanganan nilai yang tidak relevan
   - Konversi format tanggal

4. **Feature Engineering**
   - Pembuatan fitur waktu (jam, hari, bulan transaksi)
   - Perhitungan umur pengguna
   - Perhitungan jarak antara pengguna dan merchant
   - Konversi variabel kategori menjadi numerik

5. **Analisis Lanjutan & Visualisasi**
   - Visualisasi distribusi jumlah transaksi
   - Analisis pola berdasarkan waktu
   - Analisis korelasi antar fitur

6. **Persiapan Data untuk Modeling**
   - Penghapusan fitur redundant
   - Standarisasi fitur numerik

7. **Train-Test Split**
   - Pembagian data menjadi 80% training dan 20% testing

8. **Feature Engineering Lanjutan**
   - Perhitungan fraud rate per kategori berdasarkan data training
   - Penerapan transformasi yang aman tanpa kebocoran data

9. **Model Training & Evaluasi**
   - Random Forest dengan parameter default
   - XGBoost dengan optimasi hyperparameter
   - Evaluasi performa model

10. **Model Stacking**
    - Kombinasi model-model terbaik
    - Evaluasi performa model gabungan

11. **Analisis Feature Importance**
    - Identifikasi fitur yang paling berpengaruh
    - Visualisasi fitur penting

12. **Perbandingan Model & Kesimpulan**
    - Perbandingan performa antar model
    - Penentuan model terbaik

## Hasil

Perbandingan performa model (ROC AUC):

| Model | ROC AUC |
|-------|---------|
| Random Forest | 0.9925 |
| XGBoost (Tuned) | 0.9973 |
| Stacking | 0.9979 |

Model Stacking memberikan hasil terbaik dengan ROC AUC 0.9979, menunjukkan kemampuan yang sangat baik dalam membedakan transaksi normal dan fraud.

## Cara Penggunaan

### Kebutuhan Sistem

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

### Menjalankan Kode

1. Upload dataset `fraudTrain.csv` ke Google Drive
2. Mount Google Drive di Google Colab
3. Jalankan kode secara berurutan dari tahap 1 sampai 12

## Tuning Hyperparameter

Untuk model XGBoost, optimasi parameter dilakukan dengan RandomizedSearchCV dengan parameter:
- n_estimators: [50, 100]
- learning_rate: [0.01, 0.1]
- max_depth: [3, 5]
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]
- scale_pos_weight: [1, weight_ratio] (untuk menangani ketidakseimbangan kelas)

## Catatan Penting

1. **Mengatasi Warning Worker Timeout**
   - Jika muncul peringatan "A worker stopped while some jobs were given to the executor", coba:
     - Kurangi ukuran dataset untuk tuning
     - Kurangi jumlah iterasi dan parameter
     - Kurangi paralelisme dengan n_jobs=2 
     - Gunakan parameter XGBoost yang lebih hemat memori

2. **Menangani Ketidakseimbangan Kelas**
   - Data fraud biasanya jauh lebih sedikit dibanding non-fraud
   - Stratified sampling digunakan untuk mempertahankan distribusi kelas
   - Parameter scale_pos_weight disesuaikan pada model XGBoost untuk memberikan bobot lebih pada kelas minoritas

## Langkah Selanjutnya

Beberapa pengembangan yang bisa dilakukan:
- Optimasi threshold untuk menyeimbangkan precision dan recall
- Implementasi model ke dalam sistem produksi
- Pengembangan dashboard monitoring performa model
- Analisis mendalam terhadap faktor-faktor penyebab fraud

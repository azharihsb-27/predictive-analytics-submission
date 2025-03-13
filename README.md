# Laporan Proyek Machine Learning - M Kasim Azhari Hasibuan

## Domain Proyek

### Latar Belakang

Industri asuransi kesehatan memiliki peran penting dalam menyediakan perlindungan finansial terhadap risiko biaya medis yang tidak terduga. Dalam sistem asuransi, penentuan premi menjadi tantangan utama bagi perusahaan. Premi asuransi biaya biasanya dihitung berdasarkan beberapa faktor, seperti usia, jenis kelamin, index massa tubuh (BMI), kebiasaan merokok, jumlah tanggungan, dan wilayah tempat tinggal. Kesalahan dalam menetapkan biaya medis dapat menyebabkan kerugian finansial bagi perusahaan atau beban yang tidak wajar bagi pelanggan.

Seiring berkembangnya teknologi, pendekatan berbasis machine learning (ML) menjadi solusi yang baik untuk meningkatkan akurasi prediksi biaya medis. Dengan memanfaatkan data historis, model ML dapat mengidentifikasi pola-pola kompleks. Implementasi model ini dapat membantu perusahaan asuransi dalam menetapkan premi yang lebih akurat, mengurangi risiko finansial, serta memberikan manfaat bagi pelanggan dengan menawarkan tarif yang lebih sesuai.

Dalam proyek ini, saya membangun model prediksi biaya asuransi dengan menggunakan teknik machine learning. Model ini diharapkan dapat mengoptimalkan estimasi biaya medis berdasarkan faktor-faktor yang relevan yang tersedia.

---

Mengapa masalah ini harus diselesaikan?
1. Premi yang tidak akurat: Kesalahan prediksi biaya dapat menyebabkan premi yang terlalu tinggi bagi pelanggan atau terlalu rendah bagi perusahaan asuransi. Hal ini berpotensi menimbulkan kerugian finansial bagi kedua pihak.
2. Dampak finansial bagi pihak perusahaan: Pihak perusahaan akan mengalami defisit akibat pembayaran klaim yang lebih besar dari premi yang dikumpulkan karena prediksi yang tidak akurat.
3. Ketidakadilan bagi para pelanggan: Premi yang tidak sesuai dapat membuat pelanggan membayar lebih mahal atau tidak mendapat perlindungan yang layak.

Bagaimana masalah ini harus diselesaikan?
1. Pengumpulan data: Proyek ini menggunakan dataset yang tersedia di platform kaggle, yaitu [medical_insurance_cost](https://www.kaggle.com/datasets/mirichoi0218/insurance)
2. Eksplorasi dan analisis data: Melakukan analisa statistik dan visualisasi untuk memahami pola yang berpengaruh terhadap kenaikan biaya. Kemudian, mengidentifikasi korelasi antar variabel.
3. Pemilihan dan pelatihan model: Menggunakan algoritma machine learning berjenis regresi untuk membangun model prediksi. Pada proyek ini algoritma yang digunakan adalah XGBoost serta menggunakan metrik mean squarred error (MSE).
4. Optimasi dan evaluasi model: Setelah model selesai dilatih, model akan dievaluasi untuk melihat kekurangan dari model. Kemudian akan dilakukan hyperparameter tuning untuk meningkatkan kinerja model.

Referensi jurnal: [A Computational Intelligence Approach for Predicting MedicalInsurance Cost](https://onlinelibrary.wiley.com/doi/epdf/10.1155/2021/1162553)

## Business Understanding

### Problem Statements

- Ketidakakuratan prediksi biaya asuransi: Metode internasional sering tidak akurat dalam memprediksi biaya asuransi medis, sehingga menyebabkan penetapan premi yang tidak sesuai.
- Identifikasi faktor dominan: Tidak semua faktor memiliki pengaruh yang sama terhadap kenaikan biaya asuaransi medis, sehingga analisis untuk mengidentifikasi faktor yang berpengaruh sangat diperlukan.
- Optimasi model XGBoost: Diperlukan hyperparameter tuning untuk meningkatkan performa model dalam memprediksi biaya asuransi medis agar lebih akurat

### Goals

- Meningkatkan akurasi prediksi: Mengembangkan model berbasis XGBoost untuk meningkatkan keakuratan prediksi biaya asuransi medis dibandingkan metode tradisional.
- Menganalisis faktor yang berpengaruh: Mengidenfitikasi fitur atau kolom yang memiliki dampak yang signifikan terhadap kenaikan biaya asuransi medis, seperti usia, BMI, kebiasaan merokok, dan wilayah tempat tinggal.
- Mengoptimalkan model XGBoost: Melakukan hyperparameter tuning untuk meningkatkan kinerja model dan memastikan prediksi yang lebih akurat dari prediksi sebelumnya.

### Solution statements

- Penerapan XGBoost sebagai algoritma utama model:
   - Menggunakan XGBoost sebagai model karena kemampuannya dalam menangani hubungan non-linear dan fitur dengan tingkat kepentingan yang berbeda.
   - Melatih model dengan dataset yang telah dibersihkan dan diolah untuk mendapatkan prediksi yang lebih akurat.
   - Menggunakan metrik seperti mean squared error (MSE) untuk mengukur kinerja model.

- Optimasi model dengan hyperparameter tuning:
   - Melakukan XGBoost hyperparameter tuning seperti learning_rate, max_depth, n_estimators, dan min_child_weight untuk meningkatkan akurasi prediksi.
   - Menerapkan teknik pencarian hyperparameter grid search untuk menemukan kombinasi parameter terbaik.
   - Mengevaluasi hasil tuning dengan membandingkan kinerja model sebelum dan sesudah optimasi dengan metrik MSE.

- Pemilihan MSE sebagai metrik:
   - MSE digunakan sebagai metrik utama karena memberikan penalti yang lebih besar terhadap error besar, yang membantu meningkatkan keakurasian model.
   - MSE memiliki keunggulan dalam mengukur seberapa jauh prediksi dari nilai yang sebenarnya.
   - MSE membantu dalam menyesuaikan model agar lebih akurat dalam menangani kasus dengan error tinggi.

## Data Understanding

### Informasi Dataset

- Jumlah data: 1338 baris dan 7 kolom
- Kolom yang ada:
  - 'sex' (jenis kelamin, kategori: 'male', 'female')

     ![image-1-resized](https://github.com/user-attachments/assets/0c84d468-bfb7-4759-a928-ce5cd55e9ab0)
  - 'smoker' (status merokok, kategori: 'yes', 'no')

     ![image-2-resized](https://github.com/user-attachments/assets/3188cc0a-77d2-49f6-9aab-9321cb29f53a)
  - 'region' (wilayah tempat tinggal, kategori: 'northeast', 'northwest', 'southeast', 'southwest')

     ![image-3-resized](https://github.com/user-attachments/assets/8fc63a4f-9fd5-4d00-91e7-1a234a4a3cfc)
  - 'age' (usia, numerik)
  - 'bmi' (indeks massa tubuh, numerik)
     ![image-4](https://github.com/user-attachments/assets/b4d2ae7a-2165-489c-9e0f-ebf94c3966aa)
  - 'children' (jumlah anak, numerik)
  - 'charges' (biaya asuransi, numerik)(target)
     ![image-5](https://github.com/user-attachments/assets/29236e76-8197-473c-8ae9-30fd7a1c9914)
- Ringkasan statistik:
  - Usia: Rata-rata 39 tahun dengan rentang 18 sampai 64 tahun.
  - BMI: Rata-rata 30,66 dengan rentang 15,96 sampai 53,13.
  - Jumlah anak: Rata-rata 1 dengan rentang 0 sampai 5.
  - Biaya asuransi: Rata-rata 13270 USD dengan rentang 1121 USD sampai 63770 USD.
- Kondisi dataset:
  - Tidak ada missing value
  - Variabel kategori: 'sex', 'smoker', 'region'

Sumber data: [Medical Cost Insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance).

### Variabel-variabel pada dataset Medical Cost Insurance adalah sebagai berikut:

- charges (Target): merepresentasikan tagihan biaya rumah sakit.
- age: merepresentasikan umur.
- sex: merepresentasikan jenis kelamin.
- bmi: merepresentasikan nilai keidealan berat tubuh seseorang (18.5 ≤ BMI < 24.9 → Normal)
- children: merepresentasikan jumlah anak yang dimiliki.
- smoker: merepresentasikan perokok atau tidak (yes/no).
- region: merepresentasikan wilayah geografis tempat tinggal.

Pada proyek ini, beberapa tahapan Exploratory Data Analysis yang dilakukan adalah:

- Mendeskripsikan variebel
- Menangani missing value dan outliers
- Menganalisis univariate variable
- Menganalisis multivariate variable

## Data Preparation

Adapun beberapa teknik data preparation yang dilakukan pada proyek ini adalah:

1. Menangani outliers
2. Melakukan drop terhadap fitur 'children'
3. Encoding fitur kategori
4. Melakukan train-test split

Penjelasan data preparation yang dilakukan:
1. Menangani outliers: Menghapus seluruh data yang bernilai sangat jauh dari mayoritas nilai lainnya dalam keseluruhan dataset.
2. Melakukan drop terhadap fitur 'children': Fitur 'children' tidak begitu memiliki dampak yang besar terhadap kenaikan biaya asuransi atau target karena fitur ini memiliki korelasi yang rendah terhadap fitur target.
3. Encoding fitur kategori: Mengubah fitur kategori (sex, smoker, region) menjadi nilai numerik menggunakan teknik one-hot encoding dan label encoding agar dapat diproses oleh model.
4. Train-test split: Memisahkan data training dan data test dengan rasio 80:20 agar model dapat belajar dari sebagian besar data training dan diuji menggunakan data test yang belum pernah dilihat model sebelumnya.

Alasan mengapa diperlukan tahapan data preparation tersebut:
1. Menangani outliers: Outliers dapat menyebabkan rata-rata biaya asuransi akan menjadi sangat tinggi. Selain itu, model yang digunakan untuk melakukan prediksi bisa menjadi bias dan menghasilkan sebuah prediksi yang tidak akurat. Maka dari itu outliers ini perlu ditangani.
2. Melakukan drop terhadap fitur 'children': Fitur yang berkorelasi rendah dengan target perlu dihapus untuk meningkatkan efisiensi. Selain itu, mempertahankan fitur yang kurang relevan dapat menyebabkan overfitting.
3. Encoding fitur kategori: Model XGBoost hanya dapat memproses nilai numerik, sehingga fitur kategori harus dikonversi ke numerik agar dapat digunakan dalam proses pengembangan model.
4. Train-test split: Memastikan bahwa model diuji dengan data yang tidak digunakan selama training sehingga dapat memberikan gambaran nyata tentang kinerja model saat digunakan menggunakan data baru.

## Modeling

Tahapan dan parameter yang digunakan dalam proses pemodelan adalah sebagai berikut.

1. Inisialisasi model: Model yang digunakan dalam proyek ini adalah XGBoost Regressor, yang merupakan algoritma berbasis pohon keputusan yang dioptimalkan menggunakan boosting. Model ini dipilih karena kemampuannya dalam menangani hubungan non-linear. Adapun parameter yang digunakan dalam pengembangan model ini adalah n_estimators, learning_rate dan random_state.

2. Pelatihan model: Setelah model selesai diinisialisasi, dilakukan pelatihan menggunakan data training. Proses ini melibatkan pemberian input (X_train) dan target (y_train) ke dalam model.

3. Evaluasi awal model: Sebelum melakukan tuning, model dievaluasi menggunakan metrik mean squared error (MSE) untuk mengukur seberapa jauh nilai prediksi dengan nilai sebenarnya.

4. Hyperparameter tuning: Setelah model dievaluasi untuk yang pertama kalinya, untuk meningkatkan akurasi, dilakukan pencarian kombinasi parameter yang terbaik menggunakan grid search. Parameter yang dioptimalkan dalam proyek ini adalah n_estimators, learning_rate, max_depth, dan min_child_weight. Parameter ini memiliki pengaruh seperti berikut.
   - n_estimators: Nilai pohon yang digunakan untuk peningkatan ditentukan oleh n_estimator. Nilai yang terlalu kecil dapat menyebabkan underfitting, sementara nilai yang terlalu besar dapat meningkatkan risiko overfitting dan memperlambat proses pelatihan.
   - learning_rate: Nilai yang terlalu kecil membuat model belajar lebih stabil tetapi membutuhkan lebih banyak iterasi, sedangkan nilai yang terlalu besar dapat menyebabkan model tidak menemukan solusi yang tepat.
   - max_depth: Nilai untuk kedalaman maksimum setiap pohon dalam peningkatan. Kedalaman yang lebih besar memungkinkan model untuk menangkap lebih banyak pola dalam data, tetapi terlalu besar dapat membuat model terlalu kompleks dan cenderung overfitting.
   - min_child_weight: Sebelum pemisahan lebih lanjut, min_child_weight mengatur jumlah sampel minimum per leaf node. Nilai yang lebih besar membantu regularisasi dengan mencegah pemisahan berlebihan, tetapi nilai yang lebih kecil membuat model lebih fleksibel tetapi meningkatkan risiko overfitting.
5. Evaluasi akhir model: Setelah model dituning, model akan dilatih ulang dan melakukan prediksi ulang.

---

Kelebihan dan kekurangan dari setiap algoritma yang digunakan.
  Kelebihan XGBoost:

  1. Performa tinggi dengan kecepatan dan efisiensi dalam training.
  2. Menangani missing value dengan baik
  3. Mampu menangkap hubungan non-linear dan interaksi antar fitur

  Kekurangan:

  1. Sensitif terhadap hyperparameter, membutuhkan tuning untuk hasil yang optimal.
  2. Dapat mengalami overfit jika model terlalu kompleks.

---

Pada tahap awal, model digunakan dengan parameter sederhana. Namun, untuk meningkatkan akurasi model, dilakukan hyperparameter tuning menggunakan grid search.

  1. Model awal
     Sebelum melakukan tuning, model diinisialisasi dengan parameter sederhana seperti berikut.

     ```sh
     model = XGBRegressor(n_estimators=30, learning_rate=0.01, random_state=36)
     ```

  2. Hyperparameter tuning menggunakan grid search
     Kemudian untuk mencari parameter terbaik, grid search digunakan dengan variasi beberapa hyperparameter, seperti n_estimators, learning_rate, max_depth, dan min_child_weight. Berikut kode yang dituliskan.

     ```sh
     # Inisialisasi parameter yang akan dituning
     param_grid = {
         'n_estimators': [100, 200, 300],
         'learning_rate': [0.01, 0.1, 0.2],
         'max_depth': [3, 4, 5],
         'min_child_weight': [1, 2, 3],
     }

     # Melakukan Grid Search
     grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, verbose=2)
     grid_search.fit(X_train, y_train)
     ```

  3. Model setelah tuning
     Setelah menemukan kombinasi parameter terbaik, model dilatih kembali menggunakan parameter tersebut dan dilakukan evaluasi ulang.

     ```sh
     # Mengambil parameter terbaik
     best_params = grid_search.best_params_
     print("Best Hyperparameters:", best_params)

     # Melakukan train ulang dengan parameter baru
     final_model = XGBRegressor(**best_params)
     final_model.fit(X_train, y_train)
     ```

## Evaluation

- Metrik yang digunakan

  Metrik yang digunakan dalam proyek ini adala mean squared error (MSE). MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya dengan rumus:
  
  ![mse](https://github.com/user-attachments/assets/67c0c125-cc80-4252-92e1-b7599299e88b)

  dimana:

  - yi adalah nilai sebenarnya
  - y^i adalah nilai prediksi
  - n adalah jumlah sampel

- Formula metrik dan bagaimana metrik tersebut bekerja.

  Mean Squared Error (MSE) adalah metrik evaluasi yang digunakan untuk mengukur seberapa baik model regresi dalam memprediksi nilai kontinu. MSE menghitung rata-rata dari selisih kuadrat antara nilai aktual dan nilai.

  MSE didefinisikan dengan rumus berikut:
  
  ![mse](https://github.com/user-attachments/assets/9c2f88e4-3826-4186-921f-7e59dfcf7302)

  di mana:

  - yi adalah nilai sebenarnya
  - y^i adalah nilai prediksi
  - n adalah jumlah sampel

  Cara Kerja MSE

  - Mengukur Error: MSE menghitung selisih antara nilai sebenarnya dan nilai prediksi untuk setiap sampel dalam dataset.
  - Mengkuadratkan Error: Selisih tersebut dikuadratkan untuk memastikan bahwa error negatif dan positif tidak saling meniadakan, serta untuk memberikan penalti lebih besar pada kesalahan yang besar.
  - Menghitung Rata-rata: Semua error yang telah dikuadratkan dijumlahkan dan dibagi dengan jumlah total sampel untuk mendapatkan nilai rata-rata.

- Hasil proyek berdasarkan metrik evaluasi terhadap setiap problem statement, goals dan solution statement
  - Problem Statement
    - Ketidakakuratan prediksi biaya asuransi: Dengan menggunakan model XGBoost, prediksi biaya asuransi menjadi lebih akurat dibandingkan metode konvensional. Hal ini terlihat dari penurunan nilai Mean Squared Error (MSE) setelah dilakukan optimasi model.

      Sebelum dilakukannya optimasi, nilai mse adalah sebagai berikut.
      | train    | test     |
      | -------- | -------- |
      | 0.364104 | 0.373542 |
      
      Setelah dilakukannya optimasi, nilai mse adalah sebagai berikut.
      | train    | test     |
      | -------- | -------- |
      | 0.183073 | 0.163774 |
      
    - Identifikasi faktor dominan: Model ini juga membantu mengidentifikasi faktor-faktor yang memiliki pengaruh signifikan terhadap biaya asuransi, seperti status merokok, BMI, dan usia. Analisis ini memungkinkan perusahaan asuransi menetapkan premi berdasarkan faktor risiko yang lebih jelas.
    - Optimasi model XGBoost: Dengan menerapkan hyperparameter tuning, model dapat mencapai performa yang lebih baik, yang dibuktikan dengan peningkatan akurasi prediksi biaya asuransi.
  - Goals
    - Meningkatkan akurasi prediksi: Model awal memiliki MSE sebesar 0.373542 pada data testing, sementara setelah tuning, MSE menurun menjadi 0.163774, menunjukkan peningkatan akurasi prediksi yang signifikan.
    - Menganalisis faktor yang berpengaruh: Model ini mengidentifikasi bahwa variabel seperti smoker, BMI, dan usia memiliki pengaruh paling besar terhadap biaya asuransi. Hal ini dapat membantu perusahaan dalam pengambilan keputusan terkait strategi penetapan premi.
    - Mengoptimalkan model XGBoost: Dengan melakukan hyperparameter tuning, model menjadi lebih efisien dalam melakukan prediksi dan memiliki generalisasi yang lebih baik terhadap data baru.
  - Solution Statement
    - Penerapan XGBoost sebagai algoritma utama terbukti efektif karena model ini dapat menangani hubungan non-linear antar variabel dan menghasilkan prediksi yang lebih akurat dibandingkan metode konvensional.
    - Optimasi model dengan hyperparameter tuning berdampak positif pada peningkatan performa model. MSE yang lebih rendah setelah tuning menunjukkan bahwa model menjadi lebih akurat dalam memprediksi biaya asuransi.
    - Pemilihan MSE sebagai metrik evaluasi juga berdampak besar karena penalti terhadap error yang besar memungkinkan model untuk lebih sensitif terhadap prediksi yang tidak akurat. Hal ini mendorong model untuk memberikan prediksi yang lebih realistis bagi perusahaan asuransi.

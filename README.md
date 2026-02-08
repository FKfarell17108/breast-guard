# BreastGuard

**BreastGuard** adalah *web application* prediksi risiko kanker payudara yang memberikan estimasi **persentase risiko**, lengkap dengan **rekomendasi gaya hidup & medis awal** sebagai langkah skrining dan pencegahan.

*Web application* ini dibangun dengan model **Machine Learning (XGBoost)**.

![breastguard](loading.png)

---

## Fitur Utama

- **Prediksi risiko kanker payudara** berdasarkan input data pengguna
- Menampilkan **persentase risiko**
- Rekomendasi gaya hidup dan medis awal

---

## Instalasi & Setup

1. **Clone repo**
```bash
git clone https://github.com/FKfarell17108/breast-guard.git
cd breast-guard
```
   
2. **Buat virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependensi**
```bash
pip install -r requirements.txt
```

4. **Jalankan backend**
```bash
cd backend
flask run
```

5. **Buka di browser**
```bash
http://localhost:5000
```

![breastguard](home.png)

---

## Cara Penggunaan

1. Masukkan data pengguna seperti usia, riwayat keluarga, gaya hidup, dll.
2. Klik tombol Predict / Hitung Risiko.
3. Lihat hasil prediksi dalam bentuk persentase risiko kanker payudara.
4. Dapatkan rekomendasi awal untuk langkah skrining atau pencegahan.

![breastguard](form.png)

---

## © 2025 Farell Kurniawan

Hak cipta © 2025 Farell Kurniawan. Semua hak dilindungi undang-undang.  
Distribusi dan penggunaan kode ini diizinkan sesuai dengan ketentuan lisensi **MIT**.

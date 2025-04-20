# İşe Alımda Aday Seçimi: SVM ile Başvuru Değerlendirme

Bu proje, yazılım geliştirici pozisyonu için başvuran adayların değerlendirilmesi amacıyla bir **SVM (Support Vector Machine)** modeli kullanır. Model, adayların tecrübe ve teknik puanlarına göre tahmin yapar ve bir **FastAPI** servisi olarak sunulur.

## Özellikler

- **Kural Setleri**:
  1. **Teknik Sonuç**:
     - Tecrübesi 2 yıldan az ve sınav puanı 60’tan düşük olanlar işe alınmaz.
     - `Passed: 1`, `Failed: 0`
  2. **Destekleyici Skor**:
     - Son çalıştığı departman "Yazılım Geliştirme" ise: +40
     - Top üç üniversiteden mezunsa (BOUN, ODTÜ, İTÜ) veya bu üniversitelerden yüksek lisans yaptıysa: +20
     - Mezun olduğu bölüm (Lisans veya Yüksek Lisans) "Computer Engineering" veya "Software Engineering" ise: +30
     - Lisans veya Yüksek Lisans GPA'sı 3 ve üstü ise: +5
     - Cinsiyet Kadın ise: +5 (Pozitif ayrımcılık)

- **İşe Alım Kriterleri**:
  - 1. maddeden elenmemiş olmak.
  - 2. maddeden en az 45 puan almış olmak.

## Kurulum

1. **Gerekli Kütüphaneleri Yükleyin**:
   ```bash
   pip install -r requirements.txt

2. **.env Dosyası Oluşturun**:
 '''bash
echo DATABASE_URL=postgresql://{user}:{password}@localhost:5432/{databasename} > .env

3. **FASTAPI Uyulamasını Çalıştırın**:
 '''bash
uvicorn api_predict:app --reload

**KARAR SINIRI GRAFIGI**
![image](https://github.com/user-attachments/assets/fef913fc-3c52-449c-a915-ce5c38854e96)



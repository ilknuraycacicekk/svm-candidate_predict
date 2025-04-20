from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# FastAPI uygulamasını başlat
app = FastAPI()

# Kaydedilmiş modeli yükle
model = joblib.load("svc_predict_model.pkl")

# Veri işleme fonksiyonları
def process_input_data(input_data):
    """
    Kullanıcıdan gelen verileri işleyerek modelin anlayabileceği formata dönüştürür.
    """
    # Giriş verilerini DataFrame'e dönüştür
    df = pd.DataFrame([input_data])

    # Sayısal sütunları ölçeklendirme
    scaler = StandardScaler()
    df[["technical_score", "year_of_experience"]] = scaler.fit_transform(
        df[["technical_score", "year_of_experience"]]
    )

    return df[["technical_score", "year_of_experience"]]

# Veri giriş formatını tanımlıyorum
class PredictionInput(BaseModel):
    technical_score: float
    year_of_experience: float

# Ana sayfa
@app.get("/")
def read_root():
    return {"message": "Bu bir SVM modeli için FastAPI uygulamasıdır."}

# Tahmin API'si
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Kullanıcıdan gelen verileri işleyin
    processed_data = process_input_data(input_data.dict())

    # Model ile tahmin yap
    prediction = model.predict(processed_data)

    # Tahmin sonucunu döndür
    return {"prediction": int(prediction[0])}
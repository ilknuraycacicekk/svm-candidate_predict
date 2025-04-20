from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# Rastgelelik sabit kalsın
np.random.seed(42)

# .env dosyasını yükle
load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = create_engine(database_url)

# Veritabanından veriyi alıyorum
query = "SELECT * FROM candidates"
sqldata = pd.read_sql(query, engine)
candidate_data = pd.DataFrame(sqldata)

# Kategorik ve sayısal sütunları belirle
cat_columns = [col for col in candidate_data.columns if candidate_data[col].dtype == "object"]
num_columns = [col for col in candidate_data.columns if candidate_data[col].dtype in ["float64", "int64"]]

# NaN değerleri kontrol etme ve doldurma
def find_and_fill_nan(df):
    data = df.copy()
    for col in candidate_data.columns:
        if data[col].isnull().sum() > 0 and col in cat_columns:
            data[col].fillna("Unknown", inplace=True)
        elif data[col].isnull().sum() > 0 and col in num_columns:
            data[col].fillna(data[col].median(), inplace=True)
    return data

# Kategorik verileri dönüştürme
def transform_data_cat(df):
    data = df.copy()
    for col in cat_columns:
        dummies = pd.get_dummies(data[col], drop_first=True, prefix=col).astype(int)
        data = data.drop(columns=[col])
        data = pd.concat([data, dummies], axis=1)
    return data

# Sayısal verileri ölçeklendirme
def transform_data_num(df):
    data = df.copy()
    for col in data.columns:
        if col not in ["candidate_id", "Technic_Result"]:
            data[col] = StandardScaler().fit_transform(data[[col]])
    return data

# Veriyi işleme
transformed_data = transform_data_num(transform_data_cat(find_and_fill_nan(candidate_data)))

# Model Kurma
df = transformed_data.copy()
X = df[["technical_score", "year_of_experience"]]
y = df["Technic_Result"]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = SVC(kernel="linear")
first_model = model.fit(X_train, y_train)

# Tahmin yap ve doğruluk skorunu hesapla
y_pred = model.predict(X_test)
print("Doğruluk Skoru: ", accuracy_score(y_test, y_pred))

# Model Tuning
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["linear", "rbf"]
}

grid = GridSearchCV(SVC(), param_grid, cv=10, verbose=2).fit(X_train, y_train)
print("En iyi parametreler: ", grid.best_params_, "En iyi sonuç: ", grid.best_score_)

# Final Model
final_model = SVC(kernel=grid.best_params_["kernel"], C=grid.best_params_["C"], gamma=grid.best_params_["gamma"]).fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Skorlar
print("Final Model Doğruluk Skoru: ", accuracy_score(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Karar Sınırı Görselleştirme
X_train_2d = X_train[["technical_score", "year_of_experience"]]
X_test_2d = X_test[["technical_score", "year_of_experience"]]

# Karar sınırını çizmek için bir meshgrid oluştur
x_min, x_max = X_train_2d.iloc[:, 0].min() - 1, X_train_2d.iloc[:, 0].max() + 1
y_min, y_max = X_train_2d.iloc[:, 1].min() - 1, X_train_2d.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Her bir nokta için tahmin yap
Z = final_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Karar sınırını çiz
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# Eğitim ve test verilerini görselleştir
plt.scatter(X_train_2d.iloc[:, 0], X_train_2d.iloc[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor="k", label="Eğitim Verisi")
plt.scatter(X_test_2d.iloc[:, 0], X_test_2d.iloc[:, 1], c=y_test, cmap=plt.cm.coolwarm, edgecolor="k", marker="x", label="Test Verisi")

# Grafik ayarları
plt.title("Karar Sınırı")
plt.xlabel("Technical Score")
plt.ylabel("Year of Experience")
plt.legend()
plt.show()

# Modeli kaydet
joblib.dump(final_model, "svc_predict_model.pkl")
print("Model başarıyla kaydedildi.")
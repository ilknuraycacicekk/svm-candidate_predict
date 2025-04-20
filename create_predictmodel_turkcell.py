from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler

np.random.seed(42)  # Rastgelelik sabit kalsın 

load_dotenv()
database_url = os.getenv("DATABASE_URL")
engine = create_engine(database_url)

#veritabanından veriyi alıyorum:
query = "SELECT * FROM candidates"
sqldata = pd.read_sql(query, engine)
candidate_data=pd.DataFrame(sqldata)
#print(candidate_data.head())


cat_columns=[col for col in candidate_data.columns if candidate_data[col].dtype=="object"]
num_columns=[col for col in candidate_data.columns if candidate_data[col].dtype=="float64" 
             or candidate_data[col].dtype=="int64"]
"""
print({

    "Kategorik Değişkenler" : cat_columns,
    "Numerik Değişkenler" : num_columns
})
"""

#NaN değerleri kontrol etme  ve NaN değerlerini doldurma :
def find_and_fill_nan(df):
    data=df.copy()
    for col in candidate_data.columns:
        if data[col].isnull().sum() > 0 and col in cat_columns:
            data[col].fillna("Unknown", inplace=True)
        elif data[col].isnull().sum() > 0 and col in num_columns:
            data[col].fillna(data[col].median(),inplace=True)
        else : pass
    return data

#print(find_and_fill_nan(candidate_data).head(10))

def transform_data_cat(df):
    data=df.copy()
    for col in cat_columns:
        dummies = pd.get_dummies(data[col], drop_first=True, prefix=col).astype(int)
        data = data.drop(columns=[col])  # Orijinal sütunu kaldır
        data = pd.concat([data, dummies], axis=1)  # Yeni sütunları ekle
    return data

def transform_data_num(df):
    data=df.copy()
    for col in data.columns:
        if col=="candidate_id" or col=="Technic_Result":
            pass 
        else :
            data[col]=StandardScaler().fit_transform(data[[col]])
    return data

transformed_data = transform_data_num(transform_data_cat(candidate_data))
#print(transformed_data.head())
#print(transformed_data["Final"].value_counts())

## Model Kurma ##
"""
Veriden sadece :

- technical_score -->x1
-year_of_experience -->x2
-Technic_Result --> y (1: başarılı, 0: başarısız)

"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

df = transformed_data.copy()

X = df[["technical_score","year_of_experience"]]
y = df[["Technic_Result"]]

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = SVC(kernel="linear")
first_model = model.fit(X_train, y_train)

# Tahmin yap ve doğruluk skorunu hesapla
y_pred = model.predict(X_test)
print("Doğruluk Skoru: ", accuracy_score(y_test, y_pred)) 

## Model Tunning ## ------------
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel":["linear","rbf"]
}

grid = GridSearchCV(SVC(), param_grid,cv=10,verbose=2).fit(X_train, y_train)

print("En iyi parametreler : ", grid.best_params_,
      "En iyi sonuç:",grid.best_score_)

final_model=SVC(kernel=grid.best_params_["kernel"],C=grid.best_params_["C"],gamma=grid.best_params_["gamma"]).fit(X_train,y_train)
y_pred=final_model.predict(X_test)

##Skorlar :
print("Final Model Doğruluk Skoru: ", accuracy_score(y_test, y_pred)) 
print("Classification Report :", classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Modeli kaydet
import joblib
joblib.dump(final_model, "svc_predict_model.pkl")
print("Model başarıyla kaydedildi.")


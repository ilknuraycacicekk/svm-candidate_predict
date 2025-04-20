import pandas as pd
import numpy as np
from sqlalchemy import create_engine

np.random.seed(42)  # Rastgelelik sabit kalsın 

universities = [
    "BOUN",  # Boğaziçi Üniversitesi
    "ODTU",  # Orta Doğu Teknik Üniversitesi
    "ITU",   # İstanbul Teknik Üniversitesi
    "KOC",   # Koç Üniversitesi
    "SAB",   # Sabancı Üniversitesi
    "YTU",   # Yıldız Teknik Üniversitesi
]

#en son çalıştığı departman
last_departments = [
    "SD",   # Yazılım Geliştirme (Software Development)
    "DAI",  # Veri ve Yapay Zeka (Data & AI)
    "BA",   # Ürün ve İş Analizi (Business Analysis)
    "SEC",  # Siber Güvenlik (Security)
    "CI",   # Bulut ve Altyapı (Cloud & Infrastructure)
    "RND",  # AR-GE ve Yenilik (Research & Development)
    "QA",   # Test ve Kalite (Quality Assurance)
    "OTH"   # Diğer (Other Supporting Roles)
]


degrees = [
    "CE",    # Computer Engineering
    "SE",    # Software Engineering
    "EE",    # Electrical Engineering
    "MATH",  # Mathematics
    "STAT",  # Statistics
    "ECON",  # Economics
    "BUS",   # Business Administration
    "DSAI"   # Data Science and Artificial Intelligence
]


#deneyim süresi -- İlk başta belitmemin sebebi sağlıklı departman ataması yapabilmek
experience_years = np.random.randint(0, 11, size=500)

# departman ataması
last_departments = [np.random.choice(last_departments) if i > 0 else None for i in experience_years]

"""

1- np.random.choice : Kategorik değişken ataması
2- np.random.randint : Sürekli değişken ataması
3- np.random.uniform : Sürekli değişken ataması (uniform dağılım) --> (example : 2,15/3,37)

"""

candidate_info = {
    "candidate_id": np.random.choice(np.arange(1, 1000), size=500, replace=False),
    "sex": np.random.choice(["k", "e"], size=500),
    "university": np.random.choice(universities, size=500),
    "bachelor_degree": np.random.choice(degrees, size=500),
    "GPA": np.round(np.random.uniform(2, 4, size=500), 2),
    "master_degree_uni": np.random.choice(universities, size=500),
    "master_degree": np.random.choice(degrees, size=500),
    "GPA_master": np.round(np.random.uniform(2, 4, size=500), 2),
    "year_of_experience": experience_years,
    "last_department": last_departments,
    "technical_score": np.random.randint(0, 101, size=500)
}


candidate_data=pd.DataFrame(candidate_info)
#print(candidate_data.head(20))

# 1. Veri Etiketleme
def technic_result(df):
    data=df.copy()
    data["Technic_Result"] = data.apply(lambda x : 1 if x["year_of_experience"] >= 2 and 
                                x["technical_score"] >= 60 else 0,axis=1)
    return data

candidate_result=technic_result(candidate_data)
#print(candidate_result.head(20))

def support_score(df):
    data=df.copy()
    """
    Son çalıştığı departman "Yazılım Geliştirme" ise --> +40
    Top Üç üniversiteden mezunsa ( BOUN, ODTÜ, İTÜ) yada bu üniversitelerden YL varsa --> +20
    Mezun olduğu bölüm (YL or Lisans) "Computer Engineering" yada "Software Engineering" ise --> +30
    Lisans GPA yada Yüksek Lisans GPA 3 ve üstü ise --> +5
    Cinsiyet Kadın ise --> +5 /Pozitif ayrımcılık 

    """
    support_score=0

    for index, row in data.iterrows():
        if row["last_department"] == "SD":
            support_score += 40
        if row["university"] in ["BOUN", "ODTU", "ITU"] or row["master_degree_uni"] in ["BOUN", "ODTU", "ITU"]:
            support_score += 20
        if row["bachelor_degree"] in ["CE", "SE"] or row["master_degree"] in ["CE", "SE"]:
            support_score += 30
        if row["GPA"] >= 3 or row["GPA_master"] >= 3:
            support_score += 5
        if row["sex"] == "k":
            support_score += 5
        data.at[index, "Support_Score"] = support_score
        support_score = 0
    return data

candidate_result=support_score(candidate_result)
#print(candidate_result.head(20))

def final_candidates(df):
    data=df.copy()
    """
    İşe alınacaklar 1. maddeden elenmemiş olmak , 2. maddeden en az 45 alıyor olabilmek
    
    """

    data["Final"]=data.apply(lambda x : 1 if x["Technic_Result"]==1 and 
                             x["Support_Score"] >= 45 else 0,axis=1)
    return data

final_data=final_candidates(candidate_result)
print(len([i for i in final_data["Final"] if i == 1])) # 65 işe alım adayı

#Olusturdugum veriyi PostgreSQL'e aktarmak icin :
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# .env dosyasını yükle
load_dotenv()

# PostgreSQL bağlantı URL'sini .env'den al
database_url = os.getenv("DATABASE_URL")

# SQLAlchemy bağlantı dizesi
engine = create_engine(database_url)

# Veriyi PostgreSQL'e kaydet
final_data.to_sql("candidates", engine, if_exists="replace", index=False)

print("Veri PostgreSQL'e başarıyla aktarıldı.")




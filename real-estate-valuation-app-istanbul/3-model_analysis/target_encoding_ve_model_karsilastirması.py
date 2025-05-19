import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veride okunamayan karakterleri okutmak için encoding
df = pd.read_csv(r"C:\Users\ASUS\Desktop\Bitirme\Web Scrapping\hazir _veri_css_28600.csv", encoding="windows-1254") 

# district için target encoding (price'a göre)
district_mean_price = df.groupby('district')['price'].mean()
df['district_encoded'] = df['district'].map(district_mean_price)

# neighbor için target encoding (price'a göre)
neighbor_mean_price = df.groupby('neighbor')['price'].mean()
df['neighbor_encoded'] = df['neighbor'].map(neighbor_mean_price)

# Mahalle isminden encodingli değere geçiş yapmak için
district_encoding_dict = df.groupby('district')['price'].mean().to_dict()
neighbor_encoding_dict = df.groupby('neighbor')['price'].mean().to_dict()
df = df.drop(columns=["district", "neighbor"])


# Örnek: sadece numeric sütunlar üzerinden korelasyon matrisi hesapla
corr = df.corr(numeric_only=True)

# Matplotlib ile heatmap çizimi
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr, cmap='coolwarm')  # renk haritası

# Renk çubuğu (colorbar)
fig.colorbar(cax)

# Eksen etiketlerini ayarla
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='left')
ax.set_yticklabels(corr.columns)

# Sayısal değerleri hücrelerin içine yaz
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        value = round(corr.iloc[i, j], 2)
        ax.text(j, i, str(value), va='center', ha='center', color='black')

plt.title("Korelasyon Isı Haritası (Matplotlib)", pad=20)
plt.tight_layout()
plt.show()


# nan değerleri sildim
df = df.dropna()
df.isna().sum()


# Train - Test Datas
x = df.drop(columns=["price"])
y = df["price"]

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Train-test böl
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modeller
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor()
}

# Her modeli eğit ve test et
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"{name} → RMSE: {rmse:.2f}, R²: {r2:.4f}")



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Model sonuçlarını burada toplayacağız
results = []

# Modelleri eğit ve sonuçları kaydet
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    results.append({
        "Model": name,
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

# Sonuçları tabloya dök
results_df = pd.DataFrame(results)

# === GÖRSEL ===
fig, ax1 = plt.subplots(figsize=(10, 6))

# RMSE için bar plot
ax1.bar(results_df["Model"], results_df["RMSE"], color='tomato', label='RMSE')
ax1.set_ylabel('RMSE (Düşük daha iyi)', color='tomato')
ax1.tick_params(axis='y', labelcolor='tomato')
ax1.set_xticklabels(results_df["Model"], rotation=45)

# R² için line plot (ikincil eksen)
ax2 = ax1.twinx()
ax2.plot(results_df["Model"], results_df["R2 Score"], color='seagreen', marker='o', label='R² Skoru')
ax2.set_ylabel('R² Skoru (Yüksek daha iyi)', color='seagreen')
ax2.tick_params(axis='y', labelcolor='seagreen')

plt.title("Model Performans Karşılaştırması (RMSE ve R²)")
fig.tight_layout()
plt.show()





# Kullanıcıdan ilçe ve mahalle adı al
ilce_adi = "Kartal"
mahalle_adi = "Petroliş Mah."

# İlçe ve mahalle encoding değerlerini sözlükten çek
district_encoded = district_encoding_dict.get(ilce_adi)
neighbor_encoded = neighbor_encoding_dict.get(mahalle_adi)

# Encoding bulunamadıysa uyar
if district_encoded is None or neighbor_encoded is None:
    print("Girilen ilçe veya mahalle isimleri encoding sözlüğünde bulunamadı.")
else:
    # Yeni konut verisi oluştur
    new_data = pd.DataFrame([{
        "m2": 80,
        "total_room": 3,
        "age": 0,
        "floor": 2,
        "district_encoded": district_encoded,
        "neighbor_encoded": neighbor_encoded
    }])

    # Modeli eğit ve tahmin yap
    best_model = RandomForestRegressor()
    best_model.fit(x_train, y_train)
    predicted_price = best_model.predict(new_data)

    # Tahmini yazdır
    print(f"{ilce_adi}, {mahalle_adi} için tahmin edilen konut fiyatı: {predicted_price[0]:,.0f} TL")


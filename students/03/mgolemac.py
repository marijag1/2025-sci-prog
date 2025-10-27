
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import argparse

parser = argparse.ArgumentParser(
    description="Model za predviđanje potrošnje elektricne energije prema veličini domaćinstva.")
parser.add_argument("--predict", type=float,
                    help="Broj članova domaćinstva za predviđanje (npr. 4.5)")
args, unknown = parser.parse_known_args()

data = pd.read_csv('students/03/data/19_electricity_usage.csv')
data = data.dropna()

print("Dataset shape:", data.shape)
print("\nPrvih nekoliko redova:")
print(data.head())

#  Priprema podataka
X = data[['household_size']].values   # nezavisna varijabla
y = data['electricity_usage'].values  # zavisna varijabla

# Kreiramo i treniramo model
model = LinearRegression()
model.fit(X, y)

# predpostavka
y_pred = model.predict(X)

# Metričke vrijednosti
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# Rezultati
print("\nRezultati linearne regresije")
print("=" * 40)
print(f"Slope (koeficijent): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(
    f"\nJednadžba: electricity_usage = {model.coef_[0]:.2f} * household_size + {model.intercept_:.2f}")

# Vizualizacija
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Stvarni podaci')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regresiona linija')
plt.xlabel('Veličina domaćinstva')
plt.ylabel('Potrošnja električne energije (kWh)')
plt.title('Potrošnja električne energije vs Veličina domaćinstva')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

#  Predviđanje za novi broj članova
if args.predict is not None:
    prediction = model.predict([[args.predict]])[0]
    print(
        f"\nPredviđena potrošnja za domaćinstvo od {args.predict} članova: {prediction:.2f} kWh")
plt.savefig("graf_potrosnje.png")
plt.close()

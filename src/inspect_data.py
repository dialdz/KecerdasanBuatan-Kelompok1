import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("Informasi Dataset: ")
print("=" * 50)

print("Dataset Shape:",df.shape)
print("\nData Preview:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Info: ")
df.info()

# Cek Missing Values
print("\n" + "=" * 50)
print("Cek Missing Values: ")
print("=" * 50)
missing = df.isnull().sum()
print(missing)
print("Total Missing Values:", missing.sum())

#Cek duplikasi data
print("\n" + "=" * 50)
print("Cek Duplikasi Data")
print("=" * 50)
duplicated = df.duplicated().sum()
print("Total Duplikasi Data:", duplicated)

# Statistik Deskriptif
print("\n" + "=" * 50)
print("Statistika Deskriptif")
print("=" * 50)
print(df.describe())
print("\nStatistika Deskriptif untuk Kolom Kategorikal:")
print(df.describe(include="object"))

# Target Distribution
print("\n" + "=" * 50)
print("Distribusi Machine Failure (Target Variable)")
print("=" * 50)
target_counts = df["Machine failure"].value_counts()
target_percentages = df["Machine failure"].value_counts(normalize=True) * 100

print(target_counts)
print("\nPercentage: ")
print(target_percentages)

plt.figure(figsize=(7, 5))
target_counts.sort_index().plot(kind="bar", color=["skyblue", "salmon"], rot=0)

plt.title("Distribusi Machine Failure")
plt.xlabel("Machine Failure (0 = Normal, 1 = Failure)")
plt.ylabel("Jumlah sampel")
plt.tight_layout()
plt.show()

# Numerical Features Distribution
numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

df[numerical_features].hist(
    figsize=(12, 8), 
    bins=30
)

plt.suptitle()("Distribusi Fitur Numerik")
plt.tight_layout()
plt.show()

# Correlation Analysis
correlation_features = numerical_features + ["Machine failure"]
correlation_matrix = df[correlation_features].corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)
plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j, i, 
            f"{correlation_matrix.iloc[i, j]:.2f}", 
            ha="center", va="center", color="black"
        )
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
print("\nEDA Selesai")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# dataset masuk
df = pd.read_csv('ai4i2020.csv')
X = df.drop(columns=['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df['Machine failure'].values
categorical_cols = ['Type']
numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# mlp apapun itu
class PredictiveMaintenanceMLP(nn.Module):
    def __init__(self, input_dim):
        super(PredictiveMaintenanceMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

input_dim = X_train.shape[1]
model = PredictiveMaintenanceMLP(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
epochs = 30
loss_history = [] 

print("Loading...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    loss_history.append(epoch_loss)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

# evaluasi

# EVALUASI MODEL
model.eval()

with torch.no_grad():
    # Mengambil probabilitas prediksi
    y_prob = model(X_test_t).cpu().numpy().ravel()

    # Mengubah probabilitas menjadi kelas
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

# Mengubah target aktual menjadi array 1D
y_true = y_test.astype(int)

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)

# Menampilkan hasil
print("\n HASIL EVALUASI MLP")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

# Classification Report
print("\n CLASSIFICATION REPORT ")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Failure"],
    zero_division=0
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("\n  CONFUSION MATRIX ")
print(cm)

# Visualisasi Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Failure"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - MLP Predictive Maintenance")
plt.tight_layout()
plt.show()


# visual
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, color='blue', marker='o', linestyle='-', linewidth=2)
plt.title('Kurva Training Loss Model MLP', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Binary Cross Entropy)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('training_loss_plot.png', dpi=300)
print("\nGrafik berhasil disimpan dengan nama 'training_loss_plot.png'!")
plt.show()

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = ROOT_DIR / "models" / "baseline_nn.pth"
PREPROCESSOR_PATH = ROOT_DIR / "models" / "preprocessor.joblib"


class PredictiveMaintenanceMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_dataset(df: pd.DataFrame):
    target_col = "Machine failure"
    feature_cols = [
        col for col in df.columns
        if col not in {"UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
    ]

    X = df[feature_cols]
    y = df[target_col].astype(int)

    categorical_cols = ["Type"]
    numerical_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.5,
        random_state=42,
        stratify=y_remaining,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        preprocessor,
    )


def train_model(X_train, y_train, input_dim: int, epochs: int = 30, learning_rate: float = 0.001, batch_size: int = 64):
    model = PredictiveMaintenanceMLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg_loss = epoch_loss / len(dataset)
        history.append(avg_loss)

    return model, history


def evaluate_model(model, X_test, y_test):
    X_t = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        y_prob = model(X_t).numpy().ravel()

    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Failure"], zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": acc,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }


def save_artifacts(model, preprocessor):
    os.makedirs(ROOT_DIR / "models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor,
    ) = prepare_dataset(df)
    print(
        "Split sizes: "
        f"train={len(y_train)}, validation={len(y_val)}, test={len(y_test)}"
    )
    print(
        "Failure counts: "
        f"train={int(y_train.sum())}, "
        f"validation={int(y_val.sum())}, "
        f"test={int(y_test.sum())}"
    )
    model, history = train_model(X_train, y_train, X_train.shape[1])
    print("\nValidation results:")
    evaluate_model(model, X_val, y_val)
    print("\nFinal test results:")
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, preprocessor)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")

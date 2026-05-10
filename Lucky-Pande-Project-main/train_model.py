import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# ──────────────────────────────────────────────
# 1. Load & Prepare Data
# ──────────────────────────────────────────────
df = pd.read_csv("Iris.csv")
df.drop("Id", axis=1, inplace=True)  # Drop the ID column

X = df.drop("Species", axis=1)
y = df["Species"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature scaling (important for KNN — distance-based algorithm)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 2. Find Optimal K using Cross-Validation
# ──────────────────────────────────────────────
print("=" * 50)
print("Finding optimal K value...")
print("=" * 50)

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy")
    cv_scores.append(scores.mean())
    print(f"  K={k:2d}  ->  CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_k = list(k_range)[np.argmax(cv_scores)]
print(f"\n[OK] Best K = {best_k} with CV Accuracy = {max(cv_scores):.4f}")

# ──────────────────────────────────────────────
# 3. Train Final KNN Model with Best K
# ──────────────────────────────────────────────
print(f"\n{'=' * 50}")
print(f"Training KNN model with K={best_k}...")
print("=" * 50)

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# ──────────────────────────────────────────────
# 4. Evaluate on Test Set
# ──────────────────────────────────────────────
y_pred = knn_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Set Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print(f"\n{'=' * 50}")
print("Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ──────────────────────────────────────────────
# 5. Export Model, Scaler & Encoder to model.pkl
# ──────────────────────────────────────────────
model_data = {
    "model": knn_model,
    "scaler": scaler,
    "label_encoder": le,
    "best_k": best_k,
    "accuracy": accuracy,
    "feature_names": list(X.columns),
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 50}")
print("[OK] Model exported to model.pkl")
print("=" * 50)
print(f"   - Algorithm     : KNN (K={best_k})")
print(f"   - Test Accuracy : {accuracy * 100:.2f}%")
print(f"   - Features      : {list(X.columns)}")
print(f"   - Classes       : {list(le.classes_)}")
print(f"   - Contents      : model, scaler, label_encoder, metadata")

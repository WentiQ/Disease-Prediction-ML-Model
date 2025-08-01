import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier

# === Step 1: Load and preprocess dataset ===
df = pd.read_csv("dataset.csv")

# Expecting columns: Symptom_1 to Symptom_17 and Disease
symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]
df[symptom_cols] = df[symptom_cols].fillna('').astype(str).apply(lambda col: col.str.strip().str.lower())
df["Disease"] = df["Disease"].fillna('').str.strip()

# === Step 2: Build unique symptom vocabulary ===
symptom_set = set()
for col in symptom_cols:
    symptom_set.update(df[col].unique())

symptoms = sorted(list(symptom_set - {''}))
symptom_to_index = {symptom: i for i, symptom in enumerate(symptoms)}
num_features = len(symptoms)

# === Step 3: Convert data to one-hot vectors ===
X = []
y = []

for _, row in df.iterrows():
    vector = [0] * num_features
    for col in symptom_cols:
        symptom = row[col]
        if symptom and symptom in symptom_to_index:
            vector[symptom_to_index[symptom]] = 1
    if sum(vector) > 0 and row["Disease"]:
        X.append(vector)
        y.append(row["Disease"])

# === Step 4: Encode disease labels ===
diseases = sorted(list(set(y)))
disease_to_index = {d: i for i, d in enumerate(diseases)}
y_encoded = [disease_to_index[d] for d in y]

# === Step 5: Train model ===
model = RandomForestClassifier(n_estimators=600, random_state=42)
model.fit(X, y_encoded)

# === Step 6: Save model, labels, and symptom index ===
joblib.dump(model, "model.pkl")
joblib.dump(diseases, "labels.pkl")

with open("symptom_to_index.json", "w") as f:
    json.dump(symptom_to_index, f, indent=2)

print(f"✅ Model trained on {len(X)} samples × {num_features} symptoms")
print(f"🧬 Unique diseases: {len(diseases)}")
print("💾 model.pkl, labels.pkl, and symptom_to_index.json saved.")

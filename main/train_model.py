import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
print("Loading dataset...")
df = pd.read_csv("./main/dataset_updt.csv")

# Drop rows with missing target or essential features
required_columns = ["Area", "BHK", "Bathroom", "Furnishing", "Locality", "Parking", "Status", "Transaction", "Type", "Price"]
df = df.dropna(subset=required_columns)

# Drop 'Per_Sqft' if present
df = df.drop(columns=["Per_Sqft"], errors="ignore")

# Ensure only required columns are retained
df = df[required_columns]

# Encode categorical features
categorical_cols = ["Furnishing", "Locality", "Status", "Transaction", "Type"]
encoders = {}

print("Encoding categorical features...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save trained model
print("Saving model and encoders...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Evaluate
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"✅ Model trained successfully. R² Score: {score:.2f}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load Processed Data
# ---------------------------------------------------------
filename = "uae_rents_processed.csv"
print(f"Loading data from {filename}...")
df = pd.read_csv(filename)

# Drop any rows where critical info is missing (just in case)
df = df.dropna(subset=['Price_AED', 'Area_SqFt'])

# --- OUTLIER REMOVAL (New Step) ---
# Filtering out extreme luxury villas (>3M) and data errors (<20k)
# to make the model more accurate for "normal" properties.
print(f"Rows before filter: {len(df)}")
df = df[(df['Price_AED'] > 20000) & (df['Price_AED'] < 3000000)]
df = df[(df['Area_SqFt'] > 100) & (df['Area_SqFt'] < 15000)]
print(f"Rows after filter: {len(df)}")

# 2. Define Features
# ---------------------------------------------------------
categorical_features = ['City', 'Neighborhood', 'Type']
numeric_features = [
    'Bedrooms', 'Bathrooms', 'Area_SqFt', 
    'Is_Furnished', 'Is_Upgraded', 'Has_View', 
    'Has_Maids', 'Has_Pool', 'Is_Ejari'
]

X = df[categorical_features + numeric_features]
y = df['Price_AED']

# 3. Build the Pipeline
# ---------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Define the Model
# ---------------------------------------------------------
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# 5. Train & Evaluate
# ---------------------------------------------------------
print("Splitting data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model_pipeline.fit(X_train, y_train)

print("\n--- Model Results ---")
predictions = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: AED {mae:,.2f}")
print(f"R2 Score: {r2:.3f}")

# 6. Save the Model AND Artifacts
# ---------------------------------------------------------
print("Saving model artifacts...")
joblib.dump(model_pipeline, 'house_price_model.pkl')

# Save feature names so the App can label the chart correctly
artifacts = {
    'features': numeric_features + categorical_features,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}
joblib.dump(artifacts, 'model_artifacts.pkl')

print("\n✅ Model and Artifacts saved!")
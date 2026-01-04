import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    # 1. Load Data
    # ---------------------------------------------------------
    filename = "uae_rents_processed.csv"
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    df = df.dropna(subset=['Price_AED', 'Area_SqFt'])

    # --- OUTLIER REMOVAL (Crucial for accurate tuning) ---
    print(f"Rows before filter: {len(df)}")
    df = df[(df['Price_AED'] > 20000) & (df['Price_AED'] < 3000000)]
    df = df[(df['Area_SqFt'] > 100) & (df['Area_SqFt'] < 15000)]
    print(f"Rows after filter: {len(df)}")

    # 2. Setup Features (Same as before)
    # ---------------------------------------------------------
    categorical_features = ['City', 'Neighborhood', 'Type']
    numeric_features = [
        'Bedrooms', 'Bathrooms', 'Area_SqFt', 
        'Is_Furnished', 'Is_Upgraded', 'Has_View', 
        'Has_Maids', 'Has_Pool', 'Is_Ejari'
    ]

    X = df[categorical_features + numeric_features]
    y = df['Price_AED']

    # 3. Build the Base Pipeline
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

    # Base model for tuning
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # 4. Hyperparameter Tuning (Grid Search)
    # ---------------------------------------------------------
    print("\n--- Starting Hyperparameter Tuning (Grid Search) ---")
    print("Trying different combinations of Trees (n_estimators) and Depth...")

    # Define the "Grid" of options to try
    # UPDATE: Adjusted to find simpler models (Lower depth, higher split requirement) to reduce Overfitting
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [3, 5, 10, 20],  # Added shallow depths (3, 5) to force generalization
        'regressor__min_samples_split': [5, 10, 15] # Increased minimum samples to prevent memorizing single outliers
    }

    # Run the search (CV=5 means 5-Fold Cross Validation)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X, y)

    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Cross-Validation R2 Score: {grid_search.best_score_:.3f}")

    # Update our pipeline with the best model found
    best_model = grid_search.best_estimator_

    # 5. Bias-Variance Analysis (Learning Curves)
    # ---------------------------------------------------------
    print("\n--- Generating Learning Curves (Bias/Variance Analysis) ---")

    # Calculate scores for different training set sizes (10% to 100%)
    train_sizes, train_scores, test_scores = learning_curve(
        best_model, 
        X, 
        y, 
        cv=5, 
        scoring='r2',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate means and standard deviations
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 6. Plotting the Bias-Variance Curve
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.title("Bias-Variance Learning Curve (Random Forest)")
    plt.xlabel("Training Examples (Sample Size)")
    plt.ylabel("R2 Score (Accuracy)")

    # Plot Training Score (Red)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")

    # Add labels for Training Score
    for i, txt in enumerate(train_scores_mean):
        plt.annotate(f"{txt:.2f}", (train_sizes[i], train_scores_mean[i]), 
                     textcoords="offset points", xytext=(0, 5), ha='center', color='red', fontsize=8)

    # Plot Cross-Validation Score (Green)
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Add labels for CV Score
    for i, txt in enumerate(test_scores_mean):
        plt.annotate(f"{txt:.2f}", (train_sizes[i], test_scores_mean[i]), 
                     textcoords="offset points", xytext=(0, -15), ha='center', color='green', fontsize=8)

    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()

    # SAVE instead of SHOW (Fixes the crash)
    plot_filename = 'bias_variance_plot.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Analysis Complete. Plot saved as '{plot_filename}'. Open this file to view the curve.")

    print("INTERPRETATION:")
    print("1. High Training Score (Red) + Low CV Score (Green) = High Variance (Overfitting).")
    print("2. Low Training Score + Low CV Score = High Bias (Underfitting).")
    print("3. Convergence (Lines meeting at high score) = Good Model.")

    # 7. Save the BEST model (Overwriting the old one)
    print("Saving model artifacts...")
    joblib.dump(best_model, 'house_price_model.pkl')

    # Save the feature names so the App can label the chart correctly
    artifacts = {
        'features': numeric_features + categorical_features,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    joblib.dump(artifacts, 'model_artifacts.pkl')

    print("\n✅ Best Tuned Model and Artifacts saved to 'house_price_model.pkl'")

if __name__ == "__main__":
    main()
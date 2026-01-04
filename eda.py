import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
filename = "uae_rents.csv"
print(f"Loading {filename}...")
df = pd.read_csv(filename)

# 2. Basic Cleaning (Required for plots)
# We need to turn the text columns into actual numbers to plot them
cols_to_clean = ['Price_AED', 'Bedrooms', 'Bathrooms', 'Area_SqFt']
for col in cols_to_clean:
    # 'coerce' turns non-numbers into NaN (missing values)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows where Price is missing (can't analyze what doesn't exist)
original_count = len(df)
df = df.dropna(subset=['Price_AED'])
print(f"Removed {original_count - len(df)} rows with missing prices.")

# 3. Exploration Text
print("\n--- Data Overview ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe().round(2))

print("\n--- Check for Outliers (Top 5 most expensive) ---")
print(df.nlargest(15, 'Price_AED')[['Title', 'Price_AED', 'Location']])

# 4. Visualizations
sns.set_theme(style="whitegrid")

# Plot 1: Price Distribution (Histogram)
# This helps us see if most rents are cheap or expensive
plt.figure(figsize=(10, 6))
sns.histplot(df['Price_AED'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Rental Prices (AED)')
plt.xlabel('Price (AED)')
plt.show()

# Plot 2: Price vs Bedrooms (Boxplot)
# This helps us see the price range for 1-bed, 2-bed, etc.
plt.figure(figsize=(10, 6))
sns.boxplot(x='Bedrooms', y='Price_AED', data=df)
plt.title('Rent Price Range by Bedroom Count')
plt.show()

# Plot 3: Area vs Price (Scatter)
# This shows the correlation between size and cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Area_SqFt', y='Price_AED', data=df, hue='Type', alpha=0.7)
plt.title('Price vs Area (SqFt)')
plt.show()

print("\nPlots generated. Close the windows to continue.")
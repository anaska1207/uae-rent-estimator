import pandas as pd
import numpy as np
import re

# 1. Load Data
# ---------------------------------------------------------
filename = "uae_rents.csv"
print(f"Loading {filename}...")
df = pd.read_csv(filename)

# Basic numeric cleaning
numeric_cols = ['Price_AED', 'Bedrooms', 'Bathrooms', 'Area_SqFt']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop only rows where Price is completely missing
df = df.dropna(subset=['Price_AED'])

# 2. Location Parsing (The "Divide and Conquer" Strategy)
# ---------------------------------------------------------
print("Parsing Locations (City vs. Neighborhood)...")

def parse_location(loc_str):
    if not isinstance(loc_str, str):
        return "Unknown", "Unknown", "Unknown"
    
    # Split by comma
    parts = [p.strip() for p in loc_str.split(',')]
    
    # Logic to handle different lengths
    if len(parts) == 1:
        # "Dubai"
        return parts[0], "Unknown", "Unknown"
    elif len(parts) == 2:
        # "Dubai Marina, Dubai"
        return parts[-1], parts[-2], "Unknown"
    else:
        # "Princess Tower, Dubai Marina, Dubai"
        # City is always last. Neighborhood is usually 2nd to last.
        city = parts[-1]
        neighborhood = parts[-2]
        # Everything before the neighborhood is the "Sub-community" or "Building"
        property_name = " - ".join(parts[:-2])
        return city, neighborhood, property_name

# Apply the function and create new columns
loc_data = df['Location'].apply(lambda x: pd.Series(parse_location(x)))
df['City'] = loc_data[0]
df['Neighborhood'] = loc_data[1]
df['Property_Name'] = loc_data[2]

# 3. Regex Feature Extraction (Adding "Intelligence")
# ---------------------------------------------------------
print("Extracting features from Titles...")

def engineer_features(data):
    # Initialize with 0
    data['Is_Furnished'] = 0
    data['Is_Upgraded'] = 0
    data['Has_View'] = 0
    data['Has_Maids'] = 0
    data['Has_Pool'] = 0
    data['Is_Ejari'] = 0 # New flag for low-cost/admin listings
    
    # Helper for case-insensitive regex
    def has_keyword(text, pattern):
        if not isinstance(text, str): return False
        return bool(re.search(pattern, text, re.IGNORECASE))

    # Iterate rows (Cleaner for complex logic than pure pandas sometimes)
    for index, row in data.iterrows():
        title = str(row['Title'])
        
        # Furnished: Look for 'furnished' but NOT 'unfurnished'
        if has_keyword(title, 'furnished') and not has_keyword(title, 'unfurnished'):
            data.at[index, 'Is_Furnished'] = 1
            
        # Upgraded / Renovated / Brand New
        if has_keyword(title, r'upgraded|renovated|brand new|modern|luxury'):
            data.at[index, 'Is_Upgraded'] = 1
            
        # Views (Sea, Marina, Lake, Burj, Park)
        if has_keyword(title, r'view|sea|marina|lake|creek|burj|skyline|palm'):
            data.at[index, 'Has_View'] = 1
            
        # Amenities
        if has_keyword(title, r'maid'):
            data.at[index, 'Has_Maids'] = 1
        if has_keyword(title, r'pool'):
            data.at[index, 'Has_Pool'] = 1
            
        # Special "Ejari" case (likely lower price)
        if has_keyword(title, r'ejari'):
            data.at[index, 'Is_Ejari'] = 1
            
    return data

df = engineer_features(df)

# 4. Save Processed Data
# ---------------------------------------------------------
output_filename = "uae_rents_processed.csv"
# Reorder columns to put the most interesting ones first
cols = ['Price_AED', 'City', 'Neighborhood', 'Property_Name', 
        'Type', 'Bedrooms', 'Bathrooms', 'Area_SqFt', 
        'Is_Furnished', 'Is_Upgraded', 'Has_View', 'Has_Maids', 'Has_Pool', 'Is_Ejari']

# Save only the columns we care about
df[cols].to_csv(output_filename, index=False)

print(f"\n✅ Success! Processed data saved to '{output_filename}'")
print("\n--- Preview of New Location Columns ---")
print(df[['Location', 'City', 'Neighborhood', 'Property_Name']].head(10))

print("\n--- Preview of Extracted Features ---")
print(df[['Title', 'Is_Furnished', 'Has_View', 'Is_Ejari']].head(5))
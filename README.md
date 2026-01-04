# UAE Real Estate Price Estimator (End-to-End ML Project)

https://uae-rent-estimator.streamlit.app

# Project Overview

This project is a complete Machine Learning pipeline designed to estimate residential rental prices in the United Arab Emirates.

Standard price estimators often fail because they rely solely on structured data (Bedrooms, Bathrooms, Area). However, real estate value is often hidden in unstructured text (e.g., "Sea View", "Chiller Free", "Upgraded Interiors").

This project solves that problem by:

Scraping live, unstructured listing data from the web.

Engineering Features using Regex to quantify qualitative amenities.

Predicting Prices using a Hyperparameter-tuned Random Forest model.

# Project Structure

scraper.py

Data Collection: Scrapes live listing data (Price, Location, Title, etc.) using BeautifulSoup and Requests.

process_data.py

Preprocessing: Cleans raw data, handles missing values, and extracts features from text using Regex.

train_model.py

Model Training: Trains the Random Forest model, performs Grid Search for tuning, and saves the .pkl files.

app.py

Frontend: The Streamlit application that loads the model and provides the user interface.

uae_rents_processed.csv

Data: The cleaned dataset used for training and dropdown menus.

# Technical Architecture

Data Collection: Custom web scraper with anti-bot handling to aggregate live listings.

Tech: Requests, BeautifulSoup

Data Cleaning: "Divide & Conquer" location parsing and outlier removal (filtering luxury villas > 3M AED).

Tech: Pandas, NumPy

Feature Engineering: Extracted binary flags for Furnished, Sea View, Maids Room, and Private Pool.

Tech: Regex

Model: Random Forest Regressor with GridSearchCV for hyperparameter optimization.

Tech: Scikit-Learn

Deployment: Interactive web dashboard for real-time inference.

Tech: Streamlit

# Model Performance & Diagnosis

The "Small Data" Challenge

Initial training on a small dataset (~250 rows) resulted in a High Variance (Overfitting) problem. The model memorized the training data ($R^2 \approx 0.98$) but struggled to generalize ($R^2 \approx -0.1$).

The Solution: Bias-Variance Analysis

I performed a diagnostic analysis using Learning Curves.

Diagnosis: The large gap between Training and Validation scores confirmed that the model complexity was too high for the dataset size.

Action:

Restricted tree depth (max_depth=20).

Increased split requirements (min_samples_split=10).

Implemented robust outlier filtration.

Result: Improved generalization with a test set $R^2$ of ~0.53-0.65 (Excellent for a small dataset with high human variance).

# Key Insights (Feature Importance)

The model identified the following as the primary drivers of rental price in the UAE:

Area (Sq. Ft) - The strongest predictor of price.

Neighborhood - High tier areas (e.g., Palm Jumeirah) command significant premiums over similar units elsewhere.

Is_Furnished - Furnished units showed a consistent price markup.

Bedrooms/Bathrooms - Standard correlation.

# How to Run Locally

Clone the repository

git clone https://github.com/anaska1207/uae-rent-estimator.git


Create a Virtual Environment (Recommended)

Windows
python -m venv venv
.\venv\Scripts\activate

Mac/Linux
python3 -m venv venv
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Run the App

streamlit run app.py


# Future Improvements

Scale Data Collection: Scaling the scraper to 10,000+ listings would eliminate the remaining variance gap.

NLP Analysis: Using TF-IDF or Word2Vec on property descriptions to capture sentiment ("Cozy", "Luxury", "Urgent").

Time Series: Tracking price changes over time to predict market trends.
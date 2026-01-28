import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors

# Ensure clean output encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    return pd.read_csv(DATA_PATH)

def train_regression(df):
    print("\nTraining Regression Model (Gradient Boosting)...")
    
    feature_cols = ['VisitYear', 'VisitMonth', 'VisitModeId', 'AttractionTypeId', 'CountryId', 'CityId', 'AttractionCityId']
    features = [c for c in feature_cols if c in df.columns]
    
    X = df[features].fillna(0)
    y = df['Rating']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Regression MAE: {np.mean(np.abs(y_test - preds))}")
    print(f"Regression Squared Error: {mse}")
    
    joblib.dump(model, os.path.join(MODELS_DIR, "regression_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "regression_scaler.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "regression_features.pkl"))
    print("Regression model saved.")

def train_classification(df):
    print("\nTraining Classification Model (Gradient Boosting)...")
    
    target_col = 'VisitModeId'
    if target_col not in df.columns:
        print("VisitModeId not found, skipping classification.")
        return

    feature_cols = ['VisitYear', 'VisitMonth', 'AttractionTypeId', 'CountryId', 'CityId']
    features = [c for c in feature_cols if c in df.columns]
    
    X = df[features].fillna(0)
    y = df[target_col]
    
    if y.nunique() < 2:
        print("Not enough classes for classification.")
        return

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Classification Accuracy: {acc}")
    print(classification_report(y_test, preds))
    
    joblib.dump(model, os.path.join(MODELS_DIR, "classification_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "classification_scaler.pkl"))
    joblib.dump(features, os.path.join(MODELS_DIR, "classification_features.pkl"))
    print("Classification model saved.")

def train_recommendation(df):
    print("\nTraining Recommendation System (Collaborative Filtering)...")
    # Same as before, KNN works well enough for this scope.
    valid_ratings = df[['UserId', 'AttractionId', 'Rating']].dropna()
    
    try:
        user_item_matrix = valid_ratings.pivot_table(index='UserId', columns='AttractionId', values='Rating').fillna(0)
        print(f"User-Item Matrix Shape: {user_item_matrix.shape}")
        
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(user_item_matrix)
        
        joblib.dump(model_knn, os.path.join(MODELS_DIR, "recommendation_knn.pkl"))
        joblib.dump(user_item_matrix, os.path.join(MODELS_DIR, "user_item_matrix.pkl"))
        print("Recommendation model saved.")
        
    except Exception as e:
        print(f"Error creating pivot table: {e}")

def main():
    if not os.path.exists(DATA_PATH):
        print("Data not found!")
        return
        
    df = pd.read_csv(DATA_PATH)
    train_regression(df)
    train_classification(df)
    train_recommendation(df)

if __name__ == "__main__":
    main()

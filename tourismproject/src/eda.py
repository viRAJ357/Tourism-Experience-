import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure clean output encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_data.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

def run_eda():
    if not os.path.exists(DATA_PATH):
        print("Master data not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded master data: {df.shape}")
    
    # 1. Rating Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Rating', data=df)
    plt.title('Distribution of Ratings')
    plt.savefig(os.path.join(REPORT_DIR, 'rating_distribution.png'))
    print("Saved rating_distribution.png")
    
    # 2. Visit Mode Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(y='VisitMode', data=df, order=df['VisitMode'].value_counts().index)
    plt.title('Distribution of Visit Modes')
    plt.savefig(os.path.join(REPORT_DIR, 'visit_mode_distribution.png'))
    print("Saved visit_mode_distribution.png")
    
    # 3. Top 10 Countries of Users
    if 'Country' in df.columns:
        plt.figure(figsize=(10, 8))
        top_countries = df['Country'].value_counts().head(10)
        sns.barplot(y=top_countries.index, x=top_countries.values)
        plt.title('Top 10 User Countries')
        plt.savefig(os.path.join(REPORT_DIR, 'top_countries.png'))
        print("Saved top_countries.png")
        
    # 4. Ratings by Visit Mode
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='VisitMode', y='Rating', data=df)
    plt.title('Ratings by Visit Mode')
    plt.savefig(os.path.join(REPORT_DIR, 'ratings_by_visit_mode.png'))
    print("Saved ratings_by_visit_mode.png")
    
    print("EDA Completed.")

if __name__ == "__main__":
    run_eda()

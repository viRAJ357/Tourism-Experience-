import pandas as pd
import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

data_dir = r"c:\Users\nikhi\OneDrive\Desktop\tourismproject\data\Tourism Dataset"
files = [
    "Transaction.xlsx", "User.xlsx", "City.xlsx", "Type.xlsx", 
    "Mode.xlsx", "Continent.xlsx", "Country.xlsx", "Region.xlsx", "Item.xlsx"
]

report_path = r"c:\Users\nikhi\OneDrive\Desktop\tourismproject\inspection_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        f.write(f"\n{'='*20} Inspecting {filename} {'='*20}\n")
        if not os.path.exists(filepath):
            f.write(f"File not found: {filename}\n")
            continue
        try:
            df = pd.read_excel(filepath)
            f.write(f"Shape: {df.shape}\n")
            f.write("Columns:\n")
            f.write(str(df.columns.tolist()) + "\n")
            f.write("Missing Values:\n")
            missing = df.isnull().sum()[df.isnull().sum() > 0]
            if not missing.empty:
                f.write(str(missing) + "\n")
            else:
                f.write("No missing values.\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes) + "\n")
            f.write("Head:\n")
            f.write(str(df.head(3)) + "\n")
        except Exception as e:
            f.write(f"Error reading {filename}: {e}\n")

import pandas as pd
import os

data_dir = r"c:\Users\nikhi\OneDrive\Desktop\tourismproject\data\Tourism Dataset"
files = [
    "Transaction.xlsx", "User.xlsx", "City.xlsx", "Type.xlsx", 
    "Mode.xlsx", "Continent.xlsx", "Country.xlsx", "Region.xlsx", "Item.xlsx"
]

def inspect_file(filename):
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        return

    print(f"\n{'='*20} Inspecting {filename} {'='*20}")
    try:
        df = pd.read_excel(filepath)
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nMissing Values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        print("\nData Types:")
        print(df.dtypes)
        print("\nHead:")
        print(df.head(3))
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    for f in files:
        inspect_file(f)

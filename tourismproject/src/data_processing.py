import pandas as pd
import os
import sys

# Set working directory to project root if running from src
# logic to handle paths robustly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Tourism Dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def load_data():
    print("Loading datasets...")
    files = {
        "transaction": "Transaction.xlsx",
        "user": "User.xlsx",
        "city": "City.xlsx",
        "type": "Type.xlsx",
        "mode": "Mode.xlsx",
        "continent": "Continent.xlsx",
        "country": "Country.xlsx",
        "region": "Region.xlsx",
        "item": "Item.xlsx"
    }
    
    dfs = {}
    for key, filename in files.items():
        filepath = os.path.join(DATA_DIR, filename)
        try:
            dfs[key] = pd.read_excel(filepath)
            print(f"Loaded {key}: {dfs[key].shape}")
        except Exception as e:
            print(f"Error loading {key}: {e}")
            sys.exit(1)
    return dfs

def clean_data(dfs):
    print("Cleaning data...")
    user = dfs['user']
    city = dfs['city']
    
    # User: Fill missing CityId with 0 (Unknown) and convert to int
    if user['CityId'].isnull().sum() > 0:
        print(f"Refilling {user['CityId'].isnull().sum()} missing CityIds with 0")
        user['CityId'] = user['CityId'].fillna(0).astype('int64')
    
    # City: Fill missing CityName
    if city['CityName'].isnull().sum() > 0:
        print(f"Refilling {city['CityName'].isnull().sum()} missing CityNames with 'Unknown'")
        city['CityName'] = city['CityName'].fillna('Unknown')

    # Ensure consistent column naming for merges
    # Rename columns if necessary to avoid collision or for clarity, 
    # but the schema looks mostly distinct (CityId, CountryId etc).
    
    return dfs

def merge_data(dfs):
    print("Merging datasets...")
    # Start with Transaction
    master = dfs['transaction'].copy()
    
    # Join User
    master = master.merge(dfs['user'], on='UserId', how='left')
    
    # Join VisitMode (Transaction has VisitMode column which seems to be an ID based on inspection? 
    # Let's check inspection report: 'VisitMode' in Transaction is int64. 'Mode.xlsx' has 'VisitModeId' and 'VisitMode' string.
    # Wait, in Transaction, is 'VisitMode' the ID or the string?
    # Inspection says Transaction['VisitMode'] is int64. Mode['VisitModeId'] is int64.
    # So Transaction['VisitMode'] is likely the FK to Mode['VisitModeId'].
    # The Mode table has 'VisitModeId' and 'VisitMode' (the name).
    # We should rename Transaction['VisitMode'] to VisitModeId temporarily or specify left_on/right_on.
    master = master.rename(columns={'VisitMode': 'VisitModeId'})
    master = master.merge(dfs['mode'], on='VisitModeId', how='left')
    
    # Join Attraction (Item)
    master = master.merge(dfs['item'], on='AttractionId', how='left')
    
    # Join Attraction Type
    master = master.merge(dfs['type'], on='AttractionTypeId', how='left')
    
    # Join User Location info
    # User has CityId, CountryId, RegionId, ContinentId.
    # Join City
    master = master.merge(dfs['city'], on=['CityId', 'CountryId'], how='left', suffixes=('', '_City'))
    # Note: City table has CountryId too. Join on both to be safe or just CityId. 
    # Inspection: City has CityId, CityName, CountryId. User has CityId.
    
    # Join Country
    master = master.merge(dfs['country'], on=['CountryId', 'RegionId'], how='left', suffixes=('', '_Country'))
    
    # Join Region
    master = master.merge(dfs['region'], on=['RegionId', 'ContinentId'], how='left', suffixes=('', '_Region'))
    
    # Join Continent
    master = master.merge(dfs['continent'], on='ContinentId', how='left')
    
    print(f"Merged Data Shape: {master.shape}")
    return master

def main():
    dfs = load_data()
    dfs = clean_data(dfs)
    master = merge_data(dfs)
    
    output_path = os.path.join(PROCESSED_DIR, "master_data.csv")
    print(f"Saving to {output_path}...")
    master.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

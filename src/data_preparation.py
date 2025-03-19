import pandas as pd
import seaborn as sns
import sqlite3
import os

def download_and_prepare_data():
    """
    Download the penguins dataset and transform it into a SQLite database.
    """
    print("Downloading penguins dataset...")
    # Load the penguins dataset using seaborn
    penguins = sns.load_dataset("penguins").dropna()
    
    # Display information about the dataset
    print("Dataset information:")
    print(f"Shape: {penguins.shape}")
    print(f"Columns: {penguins.columns.tolist()}")
    print(f"Species distribution: \n{penguins['species'].value_counts()}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Define database path
    db_path = os.path.join(data_dir, 'penguins.db')
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    
    # Create tables according to the schema
    # Create species table
    penguins[['species']].drop_duplicates().reset_index(drop=True).to_sql(
        'species', 
        conn, 
        index=False, 
        if_exists='replace'
    )
    
    # Create islands table
    penguins[['island']].drop_duplicates().reset_index(drop=True).to_sql(
        'islands', 
        conn, 
        index=False, 
        if_exists='replace'
    )
    
    # Create sex table
    penguins[['sex']].drop_duplicates().reset_index(drop=True).dropna().to_sql(
        'sex', 
        conn, 
        index=False, 
        if_exists='replace'
    )
    
    # Create measurements table with foreign keys
    # First, create a mapping for species, islands, and sex
    species_mapping = {species: idx for idx, species in enumerate(penguins['species'].unique(), 1)}
    island_mapping = {island: idx for idx, island in enumerate(penguins['island'].unique(), 1)}
    sex_mapping = {sex: idx for idx, sex in enumerate(penguins['sex'].dropna().unique(), 1)}
    
    # Create a new DataFrame for measurements with foreign keys
    measurements = penguins.copy()
    measurements['species_id'] = measurements['species'].map(species_mapping)
    measurements['island_id'] = measurements['island'].map(island_mapping)
    measurements['sex_id'] = measurements['sex'].map(lambda x: sex_mapping.get(x) if pd.notna(x) else None)
    
    # Select only the columns we need for the measurements table
    measurements = measurements[[
        'species_id', 'island_id', 'sex_id', 
        'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'
    ]]
    
    # Save to SQL
    measurements.to_sql('measurements', conn, index=True, index_label='id', if_exists='replace')
    
    # Close connection
    conn.close()
    
    print(f"Database created successfully at {db_path}")
    return db_path

if __name__ == "__main__":
    download_and_prepare_data()
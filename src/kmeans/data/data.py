import kagglehub
import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Load the city lifestyle segmentation dataset
    """
    # Download latest version
    path = kagglehub.dataset_download("umuttuygurr/city-lifestyle-segmentation-dataset")
    
    # Load the CSV file
    import os
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        df = pd.read_csv(os.path.join(path, csv_files[0]))
        
        # Select two interesting features for clustering
        # Income and Age are good features for lifestyle segmentation
        if 'Income' in df.columns and 'Age' in df.columns:
            return df[['Income', 'Age']]
        else:
            return pd.DataFrame()
    
    return pd.DataFrame()
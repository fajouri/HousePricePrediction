import Pandas as pd
import scikit-learn as skl

def load_data(house_prices_file):
    """Load data from a CSV file into a Pandas DataFrame."""
    data = pd.read_csv(file_path)
    return data 
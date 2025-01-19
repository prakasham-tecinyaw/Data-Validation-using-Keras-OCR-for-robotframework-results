import pandas as pd

def read_csv(csv_path):
    """Reads the CSV file and returns a list of expected values."""
    df = pd.read_csv(csv_path)
    return df.iloc[:, 1].tolist()  # Get values from the second column
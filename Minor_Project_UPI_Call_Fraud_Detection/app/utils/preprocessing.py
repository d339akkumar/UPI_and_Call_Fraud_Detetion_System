import pandas as pd

def load_dataset(file):
    """
    Reads uploaded CSV file into a DataFrame.
    Handles decoding errors safely.
    """
    try:
        df = pd.read_csv(file)
        return df
    except UnicodeDecodeError:
        file.seek(0)
        df = pd.read_csv(file, encoding="latin1")
        return df

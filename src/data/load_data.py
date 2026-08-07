# Loads and concatenates the raw well-log CSV files into a single table for the lithology-classification workflow.
import pandas as pd
import os


def load_data(file_path):
    df = pd.concat(
        [pd.read_csv(os.path.join(file_path, file)) for file in os.listdir(file_path)],
        ignore_index=True,
    )
    return df

import pandas as pd
import json
from datetime import datetime

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data['wp_articles_meta'])
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m-%d-%Y')
    return df

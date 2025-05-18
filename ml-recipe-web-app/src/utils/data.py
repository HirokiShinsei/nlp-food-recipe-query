import pandas as pd
import joblib

def load_recipes(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df

def preprocess_recipes(df):
    df['tags'] = df['tags'].apply(lambda x: [t.strip() for t in x.split(',')] if isinstance(x, str) else x)
    return df

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
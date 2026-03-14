import pandas as pd
import numpy as np

def load_fpl_data(path="cleaned_fpl_data_2021-22.csv", season="2021-22"):
    """
    Loads, cleans, and returns the FPL dataset for a given season.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["name", "position", "round"])

    columns = [
        "name", "element", "team", "position", "value", "total_points",
        "minutes", "goals_scored", "assists", "clean_sheets", "yellow_cards",
        "influence", "creativity", "threat", "ict_index", "round"
    ]
    df = df[columns].dropna()

    df["round"] = df["round"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")
    df["position"] = df["position"].str.upper()
    df = df.sort_values(by=["round", "name"])

    print(f"Loaded and cleaned data for season {season}, shape: {df.shape}")
    return df
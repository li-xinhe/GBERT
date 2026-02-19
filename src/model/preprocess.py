import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib

DATA_PATH = Path("./manifesto_sentences.csv")

def load_and_prepare(min_count: int = 100):
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')  
    cat_le = LabelEncoder()
    df = df.dropna(subset=["category"]).copy()
    s = df["category"].astype(str).str.strip()
    num = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")

    df = df[num.between(101, 706)].copy()
    df["category_main"] = num[num.between(101, 706)].astype(int)
    
    df["category_id"] = cat_le.fit_transform(df["category_main"])
    
    print("num_classes =", len(cat_le.classes_))
    print("min/max =", df["category_main"].min(), df["category_main"].max())

    country_le = LabelEncoder()
    df["country_id"] = country_le.fit_transform(df["country"])

    year_le = LabelEncoder()
    df["year_id"] = year_le.fit_transform(df["year"])

    macro_cols = [
        "battle_value",
        "gdp_value",
        "trade_value",
        "debt_value",
        "inflation_value",
        "unemployment_value"
    ]

    df[macro_cols] = df[macro_cols].fillna(0.0)

    macro_means = df[macro_cols].mean()
    macro_stds = df[macro_cols].std().replace(0, 1.0)
    df[macro_cols] = (df[macro_cols] - macro_means) / macro_stds
    df[macro_cols] = df[macro_cols].astype("float32")

    counts = df["category_id"].value_counts()
    valid_categories = counts[counts >= min_count].index
    df = df[df["category_id"].isin(valid_categories)].reset_index(drop=True)


    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["category_id"]
    )

    joblib.dump(
        {
            "cat_le": cat_le,
            "country_le": country_le,
            "year_le": year_le,
            "macro_means": macro_means,
            "macro_stds": macro_stds,
            "macro_cols": macro_cols
        },
        "preprocess_meta.joblib"
    )

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    print("Saved train/val and preprocess_meta.")

if __name__ == "__main__":
    load_and_prepare()

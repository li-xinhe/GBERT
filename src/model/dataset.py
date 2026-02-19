import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import joblib

class ManifestoSentenceDataset(Dataset):
    def __init__(self, csv_path, meta_path="preprocess_meta.joblib",
                 model_name="bert-base-multilingual-cased", max_len=128):
        self.df = pd.read_csv(csv_path)
        self.meta = joblib.load(meta_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

        self.macro_cols = self.meta["macro_cols"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["sentence"])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        macro_vals = row[self.macro_cols].to_numpy(dtype="float32")
        macro = torch.from_numpy(macro_vals)  # (num_macro,)

        country_id = torch.tensor(row["country_id"], dtype=torch.long)
        year_id = torch.tensor(row["year_id"], dtype=torch.long)
        label = torch.tensor(row["category_id"], dtype=torch.long)

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "macro": macro,
            "country_id": country_id,
            "year_id": year_id,
            "labels": label
        }
        return item

class ManifestoDebugDataset(ManifestoSentenceDataset):
    """Debug dataset: only take first n samples from CSV."""
    def __init__(
        self,
        csv_path: str,
        n: int = 64,
        meta_path: str = "preprocess_meta.joblib",
        model_name: str = "bert-base-multilingual-cased",
        max_len: int = 128,
    ):
        super().__init__(
            csv_path=csv_path,
            meta_path=meta_path,
            model_name=model_name,
            max_len=max_len,
        )
        self.df = self.df.iloc[:n].reset_index(drop=True)
        print(f"[DEBUG DATA] Loaded {len(self.df)} samples from {csv_path}")


#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from model import ManifestoSentenceDataset, ManifestoDebugDataset
from model import CausalInterpretableModel
from model import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train CausalInterpretableModel on manifesto sentences.")
    p.add_argument("--train_csv", default="train.csv")
    p.add_argument("--val_csv", default="val.csv")
    p.add_argument("--meta", default="preprocess_meta.joblib")
    p.add_argument("--model_name", default="bert-base-multilingual-cased")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug_n", type=int, default=0, help="If >0, use debug datasets with only first N rows.")
    p.add_argument("--out", default="causal_nam_best.pt")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = joblib.load(args.meta)

    num_classes = len(meta["cat_le"].classes_)
    num_countries = len(meta["country_le"].classes_)
    num_years = len(meta["year_le"].classes_)
    num_macro = len(meta["macro_cols"])

    ds_kwargs = dict(meta_path=args.meta, model_name=args.model_name, max_len=args.max_len)

    if args.debug_n and args.debug_n > 0:
        train_ds = ManifestoDebugDataset(args.train_csv, n=args.debug_n, **ds_kwargs)
        val_ds = ManifestoDebugDataset(args.val_csv, n=max(1, args.debug_n // 2), **ds_kwargs)
    else:
        train_ds = ManifestoSentenceDataset(args.train_csv, **ds_kwargs)
        val_ds = ManifestoSentenceDataset(args.val_csv, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)

    model = CausalInterpretableModel(
        model_name=args.model_name,
        num_classes=num_classes,
        num_macro=num_macro,
        num_countries=num_countries,
        num_years=num_years,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                macro=batch["macro"],
                country_id=batch["country_id"],
                year_id=batch["year_id"],
            )
            loss = criterion(logits, batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # ---- Val ----
        model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []
        top3_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val"):
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    macro=batch["macro"],
                    country_id=batch["country_id"],
                    year_id=batch["year_id"],
                )
                loss = criterion(logits, batch["labels"])
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                all_labels.append(batch["labels"].cpu().numpy())
                all_preds.append(preds.cpu().numpy())

                total += batch["labels"].size(0)
                top3 = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
                top3_correct += ((top3 == batch["labels"].unsqueeze(1)).any(dim=1).sum().item())
                if logits.size(1) >= 5:
                    top5 = torch.topk(logits, k=5, dim=1).indices
                    top5_correct += ((top5 == batch["labels"].unsqueeze(1)).any(dim=1).sum().item())

        avg_val_loss = val_loss / max(1, len(val_loader))
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
        all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])

        val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) else float("nan")
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if len(all_labels) else float("nan")
        macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        ) if len(all_labels) else (float("nan"), float("nan"), None, None)

        top3_acc = top3_correct / total if total else float("nan")
        top5_acc = (top5_correct / total) if (total and num_classes >= 5) else None

        print("=" * 80)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f}")
        print(f"Val   Acc : {val_acc:.4f}")
        print(f"Macro F1  : {macro_f1:.4f}")
        print(f"Macro Prec: {macro_precision:.4f}")
        print(f"Macro Rec : {macro_recall:.4f}")
        print(f"Top-3 Acc : {top3_acc:.4f}")
        if top5_acc is not None:
            print(f"Top-5 Acc : {top5_acc:.4f}")
        print("=" * 80)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.out)
            print(f"Saved best model -> {args.out}")


if __name__ == "__main__":
    main()

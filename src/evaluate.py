"""
Evaluasi Model
===============
Metrik:
  - NER      : seqeval entity-level F1, precision, recall
  - Category : sklearn classification_report
  - Type     : classification_report

Usage:
    python src/evaluate.py \
        --model_path outputs/best_model/ \
        --test_file  data/processed/test.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report as ner_report
from sklearn.metrics import classification_report

from dataset import TransactionDataset
from model import JointTransactionModel

ID2LABEL    = {0:"O",1:"B-AMOUNT",2:"I-AMOUNT",3:"B-MERCHANT",4:"I-MERCHANT",5:"B-PAY_METHOD",6:"I-PAY_METHOD"}
ID2CATEGORY = {0:"makanan",1:"transport",2:"belanja",3:"hiburan",4:"kesehatan",5:"pendidikan",6:"tagihan",7:"pemasukan",8:"lainnya"}
ID2TYPE     = {0:"expense",1:"income"}


def evaluate(model_path: str, test_file: str, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device: {device}")

    meta_path = os.path.join(os.path.dirname(test_file), "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    model = JointTransactionModel(model_name=meta["model_name"])
    weights = os.path.join(model_path, "pytorch_model.bin")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    loader = DataLoader(TransactionDataset(test_file), batch_size=batch_size)

    ner_preds, ner_labels   = [], []
    cat_preds, cat_labels   = [], []
    type_preds, type_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            mask = batch["attention_mask"].cpu().numpy().astype(bool)
            np_ner_pred = out["ner_logits"].argmax(-1).cpu().numpy()
            np_ner_true = batch["ner_labels"].cpu().numpy()

            for pred_row, true_row, m in zip(np_ner_pred, np_ner_true, mask):
                ner_preds.append([ID2LABEL[p] for p in pred_row[m]])
                ner_labels.append([ID2LABEL[t] for t in true_row[m]])

            cat_preds.extend(out["category_logits"].argmax(-1).cpu().numpy())
            cat_labels.extend(batch["category_label"].cpu().numpy())
            type_preds.extend(out["type_logits"].argmax(-1).cpu().numpy())
            type_labels.extend(batch["type_label"].cpu().numpy())

    print("\n=== NER (entity-level) ===")
    print(ner_report(ner_labels, ner_preds))
    print("\n=== Category ===")
    print(classification_report(cat_labels, cat_preds, target_names=list(ID2CATEGORY.values())))
    print("\n=== Type ===")
    print(classification_report(type_labels, type_preds, target_names=list(ID2TYPE.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/best_model/")
    parser.add_argument("--test_file",  default="data/processed/test.json")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    evaluate(args.model_path, args.test_file, args.batch_size)

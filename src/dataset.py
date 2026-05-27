"""
PyTorch Dataset — Joint NER + Classification
"""

import json
import torch
from torch.utils.data import Dataset


class TransactionDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
            "ner_labels":     torch.tensor(item["ner_labels"],     dtype=torch.long),
            "category_label": torch.tensor(item["category_label"], dtype=torch.long),
            "type_label":     torch.tensor(item["type_label"],     dtype=torch.long),
        }

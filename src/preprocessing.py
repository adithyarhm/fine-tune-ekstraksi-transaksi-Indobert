"""
Preprocessing Pipeline
=======================
Konversi raw JSON -> format BIO-tagged siap training.

BIO Tag Schema:
    B-AMOUNT / I-AMOUNT
    B-MERCHANT / I-MERCHANT
    B-PAY_METHOD / I-PAY_METHOD
    O

Usage:
    python src/preprocessing.py \
        --input data/synthetic/transactions.json \
        --output data/processed/
"""

import json
import os
import argparse
import random
from transformers import AutoTokenizer

MODEL_NAME = "indobenchmark/indobert-base-p2"
MAX_LENGTH  = 128

LABEL2ID = {
    "O": 0,
    "B-AMOUNT": 1,    "I-AMOUNT": 2,
    "B-MERCHANT": 3,  "I-MERCHANT": 4,
    "B-PAY_METHOD": 5, "I-PAY_METHOD": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

CATEGORY2ID = {
    "makanan": 0, "transport": 1, "belanja": 2, "hiburan": 3,
    "kesehatan": 4, "pendidikan": 5, "tagihan": 6, "pemasukan": 7, "lainnya": 8
}
TYPE2ID = {"expense": 0, "income": 1}


def align_labels(tokenizer, text: str, entities: dict) -> dict:
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    labels  = ["O"] * MAX_LENGTH
    offsets = encoding["offset_mapping"]
    text_lower = text.lower()

    def tag_span(entity_text: str, tag: str):
        if not entity_text:
            return
        start_char = text_lower.find(entity_text.lower())
        if start_char == -1:
            return
        end_char = start_char + len(entity_text)
        first = True
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if e <= start_char or s >= end_char:
                continue
            labels[i] = f"B-{tag}" if first else f"I-{tag}"
            first = False

    tag_span(entities.get("amount_text", ""), "AMOUNT")
    tag_span(entities.get("merchant",    ""), "MERCHANT")
    tag_span(entities.get("pay_method",  ""), "PAY_METHOD")

    return {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "token_type_ids": encoding.get("token_type_ids", [0] * MAX_LENGTH),
        "ner_labels":     [LABEL2ID.get(l, 0) for l in labels],
    }


def process_file(input_path: str, output_dir: str, tokenizer):
    with open(input_path, encoding="utf-8") as f:
        samples = json.load(f)

    processed = []
    for s in samples:
        enc = align_labels(tokenizer, s["text"], {
            "amount_text": s.get("amount_text", ""),
            "merchant":    s.get("merchant",    ""),
            "pay_method":  s.get("pay_method",  ""),
        })
        enc["category_label"] = CATEGORY2ID[s["category"]]
        enc["type_label"]      = TYPE2ID[s["type"]]
        enc["text"]            = s["text"]
        enc["amount"]          = s["amount"]
        processed.append(enc)

    random.seed(42)
    random.shuffle(processed)
    n = len(processed)
    train_end = int(n * 0.8)
    val_end   = int(n * 0.9)

    splits = {
        "train": processed[:train_end],
        "val":   processed[train_end:val_end],
        "test":  processed[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[preprocessing] {name:5s}: {len(data)} samples -> {path}")

    meta = {
        "label2id": LABEL2ID, "id2label": ID2LABEL,
        "category2id": CATEGORY2ID, "type2id": TYPE2ID,
        "max_length": MAX_LENGTH, "model_name": MODEL_NAME,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[preprocessing] metadata.json saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/synthetic/transactions.json")
    parser.add_argument("--output", default="data/processed/")
    args = parser.parse_args()
    print(f"[preprocessing] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    process_file(args.input, args.output, tokenizer)


if __name__ == "__main__":
    main()

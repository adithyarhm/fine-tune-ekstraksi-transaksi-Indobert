"""
Inference + Post-processing
=============================
Input  : teks bebas bahasa Indonesia
Output : JSON terstruktur sesuai AI-DS-SPEC

Usage:
    python src/inference.py --text "beli kopi 25rb pakai gopay di Kopi Kenangan"
    python src/inference.py --text "gajian 5jt dari Telkom"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import re
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import JointTransactionModel

MODEL_PATH    = "outputs/best_model/"
METADATA_PATH = "data/processed/metadata.json"

ID2LABEL    = {0:"O",1:"B-AMOUNT",2:"I-AMOUNT",3:"B-MERCHANT",4:"I-MERCHANT",5:"B-PAY_METHOD",6:"I-PAY_METHOD"}
ID2CATEGORY = {0:"makanan",1:"transport",2:"belanja",3:"hiburan",4:"kesehatan",5:"pendidikan",6:"tagihan",7:"pemasukan",8:"lainnya"}
ID2TYPE     = {0:"expense",1:"income"}

AMOUNT_PATTERNS = [
    (r"(\d+(?:[.,]\d+)?)\s*juta",            lambda m: int(float(m.group(1).replace(",",".")) * 1_000_000)),
    (r"(\d+(?:[.,]\d+)?)\s*(?:rb|ribu|k)",   lambda m: int(float(m.group(1).replace(",",".")) * 1_000)),
    (r"rp\.?\s*(\d+(?:[.,\d]+))",            lambda m: int(re.sub(r"[.,]","", m.group(1)))),
    (r"(\d{1,3}(?:[.,]\d{3})+)",             lambda m: int(re.sub(r"[.,]","", m.group(1)))),
    (r"(\d+)",                                lambda m: int(m.group(1))),
]


def normalize_amount(text: str) -> int:
    t = text.lower().strip()
    for pattern, fn in AMOUNT_PATTERNS:
        m = re.search(pattern, t)
        if m:
            return fn(m)
    return 0


def extract_spans(tokens: list, labels: list) -> dict:
    spans = {"AMOUNT": [], "MERCHANT": [], "PAY_METHOD": []}
    cur_entity, cur_tokens = None, []

    for tok, lbl in zip(tokens, labels):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if lbl.startswith("B-"):
            if cur_entity:
                spans[cur_entity].append("".join(cur_tokens).replace(" ##", ""))
            cur_entity = lbl[2:]
            cur_tokens = [tok]
        elif lbl.startswith("I-") and cur_entity == lbl[2:]:
            cur_tokens.append(tok)
        else:
            if cur_entity:
                spans[cur_entity].append("".join(cur_tokens).replace(" ##", ""))
            cur_entity, cur_tokens = None, []

    if cur_entity:
        spans[cur_entity].append("".join(cur_tokens).replace(" ##", ""))

    return {
        "amount_text": " ".join(spans["AMOUNT"]),
        "merchant":    " ".join(spans["MERCHANT"]) or "",
        "pay_method":  " ".join(spans["PAY_METHOD"]) or "cash",
    }


class TransactionExtractor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        with open(metadata_path) as f:
            meta = json.load(f)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
        self.model     = JointTransactionModel(model_name=meta["model_name"])
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=self.device)
        )
        self.model.to(self.device).eval()
        print(f"[inference] ready on {self.device}")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            text, max_length=128, truncation=True,
            padding="max_length", return_tensors="pt"
        ).to(self.device)

        out    = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            token_type_ids=enc.get("token_type_ids"))
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        labels = [ID2LABEL[i] for i in out["ner_logits"][0].argmax(-1).cpu().numpy()]
        spans  = extract_spans(tokens, labels)

        cat_probs  = F.softmax(out["category_logits"][0], dim=-1)
        type_probs = F.softmax(out["type_logits"][0],     dim=-1)
        cat_id     = int(cat_probs.argmax())
        type_id    = int(type_probs.argmax())

        ner_conf   = float(out["ner_logits"][0].softmax(-1).max(-1).values.mean())
        confidence = round((ner_conf * float(cat_probs[cat_id]) * float(type_probs[type_id])) ** (1/3), 4)

        return {
            "amount":     normalize_amount(spans["amount_text"]),
            "category":   ID2CATEGORY[cat_id],
            "merchant":   spans["merchant"],
            "pay_method": spans["pay_method"],
            "note":       text,
            "type":       ID2TYPE[type_id],
            "confidence": confidence,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       required=True)
    parser.add_argument("--model_path", default=MODEL_PATH)
    args = parser.parse_args()
    result = TransactionExtractor(model_path=args.model_path).predict(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

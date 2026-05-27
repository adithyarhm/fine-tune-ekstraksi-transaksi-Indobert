"""
Joint NER + Classification Model
==================================
Encoder: IndoBERT (bert-base)
Head 1 — NER        : token-level classification (BIO, 7 label)
Head 2 — Category   : sequence classification (9 kelas)
Head 3 — Type       : binary classification (expense / income)

Loss = lambda_ner * L_ner + lambda_cat * L_category + lambda_type * L_type
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class JointTransactionModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_ner_labels: int = 7,
        num_category_labels: int = 9,
        num_type_labels: int = 2,
        dropout: float = 0.1,
        lambda_ner: float = 1.0,
        lambda_category: float = 0.5,
        lambda_type: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768 for bert-base

        self.dropout = nn.Dropout(dropout)

        # Head 1 — token-level NER
        self.ner_head      = nn.Linear(hidden, num_ner_labels)
        # Head 2 — sequence category (CLS)
        self.category_head = nn.Linear(hidden, num_category_labels)
        # Head 3 — sequence type (CLS)
        self.type_head     = nn.Linear(hidden, num_type_labels)

        self.lambda_ner      = lambda_ner
        self.lambda_category = lambda_category
        self.lambda_type     = lambda_type

        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        ner_labels=None,
        category_label=None,
        type_label=None,
    ):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_out    = self.dropout(out.last_hidden_state)        # (B, T, H)
        cls_out    = self.dropout(out.last_hidden_state[:, 0])  # (B, H)

        ner_logits      = self.ner_head(seq_out)        # (B, T, 7)
        category_logits = self.category_head(cls_out)   # (B, 9)
        type_logits     = self.type_head(cls_out)       # (B, 2)

        loss = None
        if ner_labels is not None:
            B, T, C = ner_logits.shape
            loss = (
                self.lambda_ner      * self.ner_loss_fn(ner_logits.view(B*T, C), ner_labels.view(B*T)) +
                self.lambda_category * self.cls_loss_fn(category_logits, category_label) +
                self.lambda_type     * self.cls_loss_fn(type_logits, type_label)
            )

        return {
            "loss":             loss,
            "ner_logits":       ner_logits,
            "category_logits":  category_logits,
            "type_logits":      type_logits,
        }

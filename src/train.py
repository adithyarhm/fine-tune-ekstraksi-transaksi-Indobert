"""
Training Script — HuggingFace Trainer API
==========================================
Setelah training selesai, otomatis export ke ONNX.

Usage:
    python src/train.py --config configs/train_config.yaml
    python src/train.py --config configs/train_config.yaml --skip_onnx
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import argparse
import torch
from transformers import TrainingArguments, Trainer

from dataset import TransactionDataset
from model import JointTransactionModel


class JointTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
            ner_labels=inputs["ner_labels"],
            category_label=inputs["category_label"],
            type_label=inputs["type_label"],
        )
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/train_config.yaml")
    parser.add_argument("--skip_onnx",  action="store_true", help="Skip ONNX export after training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]
    l_cfg = cfg["loss"]

    print(f"[train] model  : {m_cfg['base_model']}")
    print(f"[train] epochs : {t_cfg['num_train_epochs']}")
    print(f"[train] batch  : {t_cfg['per_device_train_batch_size']}")
    print(f"[train] fp16   : {t_cfg['fp16']}")

    train_ds = TransactionDataset(d_cfg["train_file"])
    val_ds   = TransactionDataset(d_cfg["val_file"])
    print(f"[train] train={len(train_ds)}  val={len(val_ds)}")

    model = JointTransactionModel(
        model_name=m_cfg["base_model"],
        num_ner_labels=7,
        num_category_labels=m_cfg["num_category_labels"],
        num_type_labels=m_cfg["num_type_labels"],
        dropout=m_cfg["dropout"],
        lambda_ner=l_cfg["lambda_ner"],
        lambda_category=l_cfg["lambda_category"],
        lambda_type=l_cfg["lambda_type"],
    )

    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        warmup_ratio=t_cfg["warmup_ratio"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        fp16=t_cfg["fp16"],
        bf16=t_cfg["bf16"],
        eval_strategy=t_cfg["evaluation_strategy"],
        save_strategy=t_cfg["save_strategy"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=t_cfg["greater_is_better"],
        logging_dir=t_cfg["logging_dir"],
        logging_steps=t_cfg["logging_steps"],
        seed=t_cfg["seed"],
        report_to="tensorboard",
    )

    trainer = JointTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("[train] Starting...")
    trainer.train()

    best_path = os.path.join(t_cfg["output_dir"], "best_model")
    trainer.save_model(best_path)
    print(f"[train] PyTorch model saved -> {best_path}")

    # Auto-export ke ONNX setelah training
    if not args.skip_onnx:
        print("\n[train] Exporting to ONNX...")
        from export_onnx import export_onnx, validate_onnx, quantize_onnx, save_tokenizer, save_label_maps
        from transformers import AutoTokenizer
        import json

        with open("data/processed/metadata.json") as f:
            meta = json.load(f)

        tokenizer  = AutoTokenizer.from_pretrained(meta["model_name"])
        onnx_dir   = os.path.join(t_cfg["output_dir"], "onnx")

        # Model ke CPU untuk export
        model.to("cpu")
        onnx_path, dummy_enc = export_onnx(model, tokenizer, onnx_dir)
        validate_onnx(model, dummy_enc, onnx_path)
        quantize_onnx(onnx_path, onnx_dir)
        save_tokenizer(tokenizer, onnx_dir)
        save_label_maps(onnx_dir)
        print(f"[train] ONNX export done -> {onnx_dir}")


if __name__ == "__main__":
    main()

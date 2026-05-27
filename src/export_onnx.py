"""
ONNX Export + Validation
=========================
Export JointTransactionModel ke format ONNX.
Output tiga file:
  onnx_export/model.onnx          <- full model
  onnx_export/model_quant.onnx    <- quantized INT8 (lebih kecil, cocok untuk server)
  onnx_export/tokenizer/          <- tokenizer files untuk dipakai di JS

Usage:
    python src/export_onnx.py \
        --model_path outputs/best_model/ \
        --output_dir onnx_export/

Validation:
    Script otomatis membandingkan output PyTorch vs ONNX (max diff < 1e-4).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

from model import JointTransactionModel

METADATA_PATH = "data/processed/metadata.json"


def export_onnx(
    model: JointTransactionModel,
    tokenizer,
    output_dir: str,
    max_length: int = 128,
):
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    # Dummy input untuk tracing
    dummy_text = "beli kopi 25rb pakai gopay di Kopi Kenangan"
    enc = tokenizer(
        dummy_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=(input_ids, attention_mask, token_type_ids),
            f=onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["ner_logits", "category_logits", "type_logits"],
            dynamic_axes={
                "input_ids":      {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "token_type_ids": {0: "batch_size", 1: "sequence"},
                "ner_logits":     {0: "batch_size", 1: "sequence"},
                "category_logits":{0: "batch_size"},
                "type_logits":    {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[export] ONNX model saved -> {onnx_path}")
    print(f"[export] File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
    return onnx_path, enc


def validate_onnx(model, enc, onnx_path: str):
    """Bandingkan output PyTorch vs ONNX. Max diff harus < 1e-4."""
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

    # PyTorch forward
    model.eval()
    with torch.no_grad():
        pt_out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # ONNX Runtime forward
    # Coba pakai CUDAExecutionProvider dulu, fallback ke CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {
        "input_ids":      input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    }
    ort_out = sess.run(None, ort_inputs)  # [ner_logits, category_logits, type_logits]

    # Validation
    checks = [
        ("ner_logits",       pt_out["ner_logits"].numpy(),      ort_out[0]),
        ("category_logits",  pt_out["category_logits"].numpy(), ort_out[1]),
        ("type_logits",      pt_out["type_logits"].numpy(),     ort_out[2]),
    ]
    all_ok = True
    for name, pt_val, ort_val in checks:
        diff = np.abs(pt_val - ort_val).max()
        status = "OK" if diff < 1e-4 else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"[validate] {name:20s} max_diff={diff:.2e}  [{status}]")

    if all_ok:
        print("[validate] All outputs match. ONNX export is valid.")
    else:
        print("[validate] WARNING: output mismatch detected!")
    return all_ok


def quantize_onnx(onnx_path: str, output_dir: str) -> str:
    """
    Dynamic INT8 quantization.
    Mengurangi ukuran model ~75% dengan penurunan akurasi minimal.
    """
    quant_path = os.path.join(output_dir, "model_quant.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8,
    )
    orig_mb  = os.path.getsize(onnx_path)  / 1e6
    quant_mb = os.path.getsize(quant_path) / 1e6
    print(f"[quantize] Original : {orig_mb:.1f} MB")
    print(f"[quantize] Quantized: {quant_mb:.1f} MB  ({100*(1-quant_mb/orig_mb):.0f}% smaller)")
    print(f"[quantize] Saved -> {quant_path}")
    return quant_path


def save_tokenizer(tokenizer, output_dir: str):
    tok_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tok_dir)
    print(f"[export] Tokenizer saved -> {tok_dir}")


def save_label_maps(output_dir: str):
    """Simpan label maps sebagai JSON untuk dikonsumsi langsung di JavaScript."""
    maps = {
        "id2ner": {
            "0": "O",
            "1": "B-AMOUNT",   "2": "I-AMOUNT",
            "3": "B-MERCHANT", "4": "I-MERCHANT",
            "5": "B-PAY_METHOD", "6": "I-PAY_METHOD",
        },
        "id2category": {
            "0": "makanan", "1": "transport", "2": "belanja",
            "3": "hiburan", "4": "kesehatan", "5": "pendidikan",
            "6": "tagihan", "7": "pemasukan", "8": "lainnya",
        },
        "id2type": {
            "0": "expense",
            "1": "income",
        },
        "max_length": 128,
    }
    path = os.path.join(output_dir, "label_maps.json")
    with open(path, "w") as f:
        json.dump(maps, f, indent=2)
    print(f"[export] Label maps saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  default="outputs/best_model/")
    parser.add_argument("--output_dir",  default="onnx_export/")
    parser.add_argument("--metadata",    default=METADATA_PATH)
    parser.add_argument("--no_quantize", action="store_true", help="Skip INT8 quantization")
    args = parser.parse_args()

    # Load metadata
    with open(args.metadata) as f:
        meta = json.load(f)

    print(f"[export] Loading model from {args.model_path}")
    model = JointTransactionModel(model_name=meta["model_name"])
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location="cpu")
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])

    # 1. Export ONNX
    onnx_path, dummy_enc = export_onnx(model, tokenizer, args.output_dir)

    # 2. Validate
    validate_onnx(model, dummy_enc, onnx_path)

    # 3. Quantize (default: ON)
    if not args.no_quantize:
        quantize_onnx(onnx_path, args.output_dir)

    # 4. Save tokenizer + label maps untuk JS
    save_tokenizer(tokenizer, args.output_dir)
    save_label_maps(args.output_dir)

    print("\n[export] Done! Files in", args.output_dir)
    print("  model.onnx         <- full FP32 model")
    if not args.no_quantize:
        print("  model_quant.onnx   <- INT8 quantized (recommended for backend)")
    print("  tokenizer/         <- tokenizer files")
    print("  label_maps.json    <- label maps untuk JS runtime")


if __name__ == "__main__":
    main()

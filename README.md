# Fine-Tune IndoBERT — Ekstraksi Transaksi NLP

Proyek fine-tuning `indobenchmark/indobert-base-p2` untuk mengekstrak entitas transaksi keuangan personal dari teks bahasa Indonesia.

## Task
Diberikan teks bebas seperti `"beli kopi 25rb pakai gopay di Kopi Kenangan"`, model menghasilkan:
```json
{
  "amount": 25000,
  "category": "makanan",
  "merchant": "Kopi Kenangan",
  "pay_method": "gopay",
  "note": "beli kopi 25rb pakai gopay di Kopi Kenangan",
  "type": "expense",
  "confidence": 0.92
}
```

## Arsitektur
Joint model dengan satu shared IndoBERT encoder + 3 head:
- **NER head** → span extraction: `amount`, `merchant`, `pay_method`
- **Classification head** → `category` (9 kelas)
- **Binary head** → `type` (expense / income)

## Struktur Direktori
```
.
├── data/
│   ├── raw/                  # Dataset mentah (NERGrit, synthetic)
│   ├── processed/            # Dataset setelah preprocessing (BIO tags)
│   └── synthetic/            # Output dari data_generation.py
├── src/
│   ├── data_generation.py    # Synthetic data generator
│   ├── preprocessing.py      # Tokenisasi + BIO tagging
│   ├── dataset.py            # PyTorch Dataset class
│   ├── model.py              # Joint NER + Classification model
│   ├── train.py              # Training loop (HuggingFace Trainer)
│   ├── evaluate.py           # Evaluasi F1, classification report
│   └── inference.py          # Inference + post-processing output
├── configs/
│   └── train_config.yaml     # Hyperparameter config
├── requirements.txt
└── README.md
```

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic training data
python src/data_generation.py --output data/synthetic/transactions.json --n 2000

# 3. Preprocessing: tokenisasi + BIO tagging
python src/preprocessing.py \
    --input data/synthetic/transactions.json \
    --output data/processed/

# 4. Training
python src/train.py --config configs/train_config.yaml

# 5. Evaluasi
python src/evaluate.py --model_path outputs/best_model/

# 6. Inference
python src/inference.py --text "beli kopi 25rb pakai gopay di Kopi Kenangan"
```

## Hardware Target
**RTX 4080 SUPER (16 GB VRAM)** + Ryzen 7 5800X
- Batch size default: 32
- Mixed precision (fp16): enabled
- Estimasi waktu training: ~15–30 menit / epoch (2K samples)

## Dataset
- **Warm-up**: `indonlp/indonlu` (NERGrit) — lihat `src/preprocessing.py`
- **Domain**: Synthetic data dari `src/data_generation.py`
- **Annotation tool**: [Label Studio](https://labelstud.io/) untuk data manual

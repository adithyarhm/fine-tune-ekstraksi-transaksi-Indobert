# Fine-Tune IndoBERT — Ekstraksi Transaksi NLP

Proyek fine-tuning `indobenchmark/indobert-base-p2` untuk mengekstrak entitas transaksi keuangan personal dari teks bahasa Indonesia. Output model berformat **ONNX** dan siap di-deploy di backend **JavaScript (Node.js)**.

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
│   ├── raw/                  # Dataset mentah
│   ├── processed/            # Dataset BIO-tagged
│   └── synthetic/            # Output data_generation.py
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py              # Training + auto ONNX export
│   ├── export_onnx.py        # Standalone ONNX exporter + quantizer
│   ├── evaluate.py
│   └── inference.py          # Python inference (testing)
├── js_runtime/
│   ├── extractor.js          # TransactionExtractor class
│   ├── index.js              # Demo runner
│   ├── test.js               # Schema validation tests
│   ├── package.json
│   └── README.md
├── onnx_export/              # Output export (auto-generated, di .gitignore)
├── configs/
│   └── train_config.yaml
├── requirements.txt
└── README.md
```

## Full Pipeline

### 1. Setup Python
```bash
conda create -n nlp-transaksi python=3.11 -y
conda activate nlp-transaksi
pip install -r requirements.txt
```

### 2. Generate Data + Training
```bash
python src/data_generation.py --n 2000
python src/preprocessing.py
python src/train.py --config configs/train_config.yaml
# => Otomatis export ONNX ke outputs/onnx/
```

### 3. Export ONNX (manual/opsional)
```bash
python src/export_onnx.py \
    --model_path outputs/best_model/ \
    --output_dir onnx_export/
# Output:
#   onnx_export/model.onnx          FP32 (~420 MB)
#   onnx_export/model_quant.onnx    INT8 (~110 MB) <- recommended
#   onnx_export/tokenizer/
#   onnx_export/label_maps.json
```

### 4. Run di Node.js
```bash
cd js_runtime
npm install
node index.js
```

## Hardware Target
**RTX 4080 SUPER (16 GB VRAM)** + Ryzen 7 5800X
- Batch size: 32, fp16: enabled
- Estimasi training: ~15–30 menit / epoch

## Model Size (ONNX)
| Format | Size | Use Case |
|---|---|---|
| `model.onnx` (FP32) | ~420 MB | Akurasi maksimal |
| `model_quant.onnx` (INT8) | ~110 MB | **Backend production** |

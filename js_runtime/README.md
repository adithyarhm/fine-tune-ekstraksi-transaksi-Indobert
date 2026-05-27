# JS Runtime — ONNX Inference (Node.js)

## Setup

```bash
cd js_runtime
npm install
```

## Prerequisite
Pastikan sudah menjalankan export ONNX dari Python terlebih dahulu:
```bash
# Dari root project
python src/export_onnx.py --model_path outputs/best_model/ --output_dir onnx_export/
```

Struktur yang dibutuhkan:
```
onnx_export/
├── model_quant.onnx      <- INT8 quantized (default dipakai)
├── model.onnx            <- FP32 full (fallback)
├── tokenizer/            <- tokenizer files
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── ...
└── label_maps.json       <- label maps
```

## Contoh Penggunaan

```bash
# Run demo
node index.js

# Run tests
node test.js
```

## Integrasi ke Express.js / Fastify

```js
const express = require('express');
const { TransactionExtractor } = require('./extractor');

const app = express();
app.use(express.json());

let extractor;

// Init sekali saat server start
(async () => {
  extractor = await TransactionExtractor.create('../onnx_export', { quantized: true });
  app.listen(3000, () => console.log('Server running on :3000'));
})();

app.post('/extract', async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: 'text is required' });
  try {
    const result = await extractor.predict(text);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
```

## Model Size
| File | Size |
|---|---|
| `model.onnx` (FP32) | ~420 MB |
| `model_quant.onnx` (INT8) | ~110 MB |

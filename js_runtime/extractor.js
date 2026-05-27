/**
 * TransactionExtractor — ONNX Runtime (Node.js)
 * ================================================
 * Load model.onnx + tokenizer, jalankan inference,
 * return JSON terstruktur sesuai AI-DS-SPEC.
 *
 * Dependency: onnxruntime-node, @xenova/transformers
 *
 * Usage:
 *   const { TransactionExtractor } = require('./extractor');
 *   const ext = await TransactionExtractor.create('../onnx_export');
 *   const result = await ext.predict('beli kopi 25rb pakai gopay di Kopi Kenangan');
 *   console.log(result);
 */

'use strict';

const ort  = require('onnxruntime-node');
const path = require('path');
const fs   = require('fs');

// ---------------------------------------------------------------------------
// Label maps (akan di-load dari label_maps.json)
// ---------------------------------------------------------------------------
let ID2NER, ID2CATEGORY, ID2TYPE, MAX_LENGTH;

// ---------------------------------------------------------------------------
// Amount normalization — handle "25rb", "5jt", "Rp 50.000", "50k"
// ---------------------------------------------------------------------------
const AMOUNT_PATTERNS = [
  { re: /(\d+(?:[.,]\d+)?)\s*juta/i,           fn: m => Math.round(parseFloat(m[1].replace(',', '.')) * 1_000_000) },
  { re: /(\d+(?:[.,]\d+)?)\s*(?:rb|ribu|k)/i,  fn: m => Math.round(parseFloat(m[1].replace(',', '.')) * 1_000) },
  { re: /rp\.?\s*(\d[\d.,]+)/i,                fn: m => parseInt(m[1].replace(/[.,]/g, ''), 10) },
  { re: /(\d{1,3}(?:[.,]\d{3})+)/,             fn: m => parseInt(m[1].replace(/[.,]/g, ''), 10) },
  { re: /(\d+)/,                                fn: m => parseInt(m[1], 10) },
];

function normalizeAmount(text) {
  if (!text) return 0;
  const t = text.toLowerCase().trim();
  for (const { re, fn } of AMOUNT_PATTERNS) {
    const m = t.match(re);
    if (m) return fn(m);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Span extraction dari BIO labels
// ---------------------------------------------------------------------------
function extractSpans(tokens, labels) {
  const spans = { AMOUNT: [], MERCHANT: [], PAY_METHOD: [] };
  let curEntity = null;
  let curTokens = [];

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const lbl = labels[i];

    // Skip special tokens
    if (['[CLS]', '[SEP]', '[PAD]'].includes(tok)) continue;

    if (lbl.startsWith('B-')) {
      if (curEntity) spans[curEntity].push(curTokens.join('').replace(/##/g, ''));
      curEntity = lbl.slice(2);
      curTokens = [tok];
    } else if (lbl.startsWith('I-') && curEntity === lbl.slice(2)) {
      curTokens.push(tok.startsWith('##') ? tok.slice(2) : ' ' + tok);
    } else {
      if (curEntity) spans[curEntity].push(curTokens.join('').replace(/##/g, ''));
      curEntity = null;
      curTokens = [];
    }
  }
  if (curEntity) spans[curEntity].push(curTokens.join('').replace(/##/g, ''));

  return {
    amount_text: spans.AMOUNT.join(' ').trim(),
    merchant:    spans.MERCHANT.join(' ').trim(),
    pay_method:  spans.PAY_METHOD.join(' ').trim() || 'cash',
  };
}

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function argmax(arr) {
  return arr.reduce((iMax, x, i, a) => (x > a[iMax] ? i : iMax), 0);
}

// ---------------------------------------------------------------------------
// TransactionExtractor class
// ---------------------------------------------------------------------------
class TransactionExtractor {
  constructor(session, tokenizer, labelMaps) {
    this.session   = session;
    this.tokenizer = tokenizer;
    this.labelMaps = labelMaps;

    ID2NER      = labelMaps.id2ner;
    ID2CATEGORY = labelMaps.id2category;
    ID2TYPE     = labelMaps.id2type;
    MAX_LENGTH  = labelMaps.max_length || 128;
  }

  /**
   * Factory method — async karena load ONNX session & tokenizer
   * @param {string} modelDir  Path ke onnx_export/ directory
   * @param {object} options   { quantized: true } untuk pakai model_quant.onnx
   */
  static async create(modelDir, options = {}) {
    const { quantized = true } = options;

    // 1. Load ONNX session
    const modelFile = quantized ? 'model_quant.onnx' : 'model.onnx';
    const modelPath = path.join(modelDir, modelFile);
    console.log(`[extractor] Loading ONNX model: ${modelPath}`);

    const sessionOptions = {
      // Gunakan CUDA jika tersedia, fallback ke CPU
      executionProviders: ['cuda', 'cpu'],
      graphOptimizationLevel: 'all',
    };
    const session = await ort.InferenceSession.create(modelPath, sessionOptions);
    console.log('[extractor] Model loaded.');

    // 2. Load tokenizer via @xenova/transformers
    //    Tokenizer files harus ada di modelDir/tokenizer/
    const { AutoTokenizer } = await import('@xenova/transformers');
    const tokenizerPath = path.join(modelDir, 'tokenizer');
    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
    console.log('[extractor] Tokenizer loaded.');

    // 3. Load label maps
    const labelMaps = JSON.parse(
      fs.readFileSync(path.join(modelDir, 'label_maps.json'), 'utf-8')
    );

    return new TransactionExtractor(session, tokenizer, labelMaps);
  }

  /**
   * Predict dari raw text
   * @param {string} text  Teks transaksi bahasa Indonesia
   * @returns {Promise<object>}  JSON sesuai AI-DS-SPEC
   */
  async predict(text) {
    // 1. Tokenize
    const enc = await this.tokenizer(text, {
      max_length: MAX_LENGTH,
      truncation: true,
      padding: 'max_length',
      return_tensors: 'np',     // return numpy-like arrays
    });

    // 2. Buat ONNX Tensors (type: int64)
    const toInt64 = (arr) => new ort.Tensor('int64',
      BigInt64Array.from(Array.from(arr).map(BigInt)), [1, arr.length]);

    const feeds = {
      input_ids:      toInt64(enc.input_ids.data),
      attention_mask: toInt64(enc.attention_mask.data),
      token_type_ids: toInt64(
        enc.token_type_ids ? enc.token_type_ids.data : new Array(MAX_LENGTH).fill(0)
      ),
    };

    // 3. Run inference
    const results = await this.session.run(feeds);
    const nerLogits      = Array.from(results.ner_logits.data);       // flat [T * 7]
    const categoryLogits = Array.from(results.category_logits.data);  // [9]
    const typeLogits     = Array.from(results.type_logits.data);       // [2]

    // 4. Decode NER (reshape [T, 7])
    const T          = MAX_LENGTH;
    const numNer     = Object.keys(ID2NER).length;
    const nerMatrix  = [];
    for (let i = 0; i < T; i++) {
      nerMatrix.push(nerLogits.slice(i * numNer, (i + 1) * numNer));
    }
    const nerPredIds = nerMatrix.map(row => argmax(row));
    const tokens     = this.tokenizer.convert_ids_to_tokens(
      Array.from(enc.input_ids.data)
    );
    const nerLabels  = nerPredIds.map(id => ID2NER[String(id)] || 'O');
    const spans      = extractSpans(tokens, nerLabels);

    // 5. Classification heads
    const catProbs  = softmax(categoryLogits);
    const typeProbs = softmax(typeLogits);
    const catId     = argmax(catProbs);
    const typeId    = argmax(typeProbs);

    // 6. Confidence: geometric mean dari top-1 prob ketiga head
    const nerMeanProb = nerMatrix
      .map(row => Math.max(...softmax(row)))
      .reduce((a, b) => a + b, 0) / T;
    const confidence = parseFloat(
      Math.cbrt(nerMeanProb * catProbs[catId] * typeProbs[typeId]).toFixed(4)
    );

    return {
      amount:     normalizeAmount(spans.amount_text),
      category:   ID2CATEGORY[String(catId)],
      merchant:   spans.merchant,
      pay_method: spans.pay_method,
      note:       text,
      type:       ID2TYPE[String(typeId)],
      confidence,
    };
  }
}

module.exports = { TransactionExtractor };

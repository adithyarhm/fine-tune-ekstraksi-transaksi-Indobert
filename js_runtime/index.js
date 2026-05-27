/**
 * Contoh penggunaan TransactionExtractor di Node.js
 * Jalankan: node js_runtime/index.js
 */

'use strict';

const { TransactionExtractor } = require('./extractor');

const MODEL_DIR = '../onnx_export';  // relatif dari js_runtime/

const TEST_CASES = [
  'beli kopi 25rb pakai gopay di Kopi Kenangan',
  'bayar PLN 150ribu via bca',
  'naik Grab 35k cash',
  'gajian 5jt dari Telkom',
  'langganan Netflix 54000 cc_visa',
  'beli obat di Apotek K24 75rb ovo',
];

(async () => {
  const extractor = await TransactionExtractor.create(MODEL_DIR, { quantized: true });

  console.log('\n=== Inference Results ===\n');
  for (const text of TEST_CASES) {
    const result = await extractor.predict(text);
    console.log(`Input : "${text}"`);
    console.log('Output:', JSON.stringify(result, null, 2));
    console.log('-'.repeat(60));
  }
})();

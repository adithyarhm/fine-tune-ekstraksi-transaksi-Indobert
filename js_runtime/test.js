/**
 * Unit test sederhana untuk TransactionExtractor
 * Jalankan: node js_runtime/test.js
 */

'use strict';

const { TransactionExtractor } = require('./extractor');

const MODEL_DIR = '../onnx_export';

const VALID_CATEGORIES = [
  'makanan', 'transport', 'belanja', 'hiburan',
  'kesehatan', 'pendidikan', 'tagihan', 'pemasukan', 'lainnya',
];
const VALID_PAY_METHODS = [
  'cash', 'bni', 'bca', 'mandiri', 'bri', 'bsi', 'cimb', 'danamon', 'permata',
  'gopay', 'ovo', 'dana', 'shopeepay',
  'cc_visa', 'cc_mastercard', 'cc_jcb', 'cc_amex',
];
const VALID_TYPES = ['expense', 'income'];

function assert(condition, msg) {
  if (!condition) throw new Error(`FAIL: ${msg}`);
  console.log(`  PASS: ${msg}`);
}

(async () => {
  const ext = await TransactionExtractor.create(MODEL_DIR, { quantized: true });

  console.log('\n=== Schema Validation Tests ===\n');

  const cases = [
    'beli kopi 25rb pakai gopay di Kopi Kenangan',
    'bayar listrik PLN 150ribu via bca',
    'gajian 5jt dari Telkom',
  ];

  for (const text of cases) {
    console.log(`Test: "${text}"`);
    const r = await ext.predict(text);

    assert(typeof r.amount === 'number' && r.amount > 0,     'amount is positive number');
    assert(VALID_CATEGORIES.includes(r.category),            `category valid: ${r.category}`);
    assert(typeof r.merchant === 'string',                   'merchant is string');
    assert(VALID_PAY_METHODS.includes(r.pay_method),        `pay_method valid: ${r.pay_method}`);
    assert(r.note === text,                                  'note equals input text');
    assert(VALID_TYPES.includes(r.type),                    `type valid: ${r.type}`);
    assert(r.confidence >= 0 && r.confidence <= 1,          `confidence in [0,1]: ${r.confidence}`);
    console.log();
  }

  console.log('All tests passed.');
})();

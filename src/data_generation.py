"""
Synthetic Transaction Data Generator
=====================================
Menghasilkan dataset teks transaksi bahasa Indonesia dengan label:
  - BIO tags untuk NER (amount, merchant, pay_method)
  - category label (9 kelas)
  - type label (expense / income)

Usage:
    python src/data_generation.py --output data/synthetic/transactions.json --n 2000
"""

import json
import random
import argparse
import os

random.seed(42)

CATEGORY_LABELS = [
    "makanan", "transport", "belanja", "hiburan",
    "kesehatan", "pendidikan", "tagihan", "pemasukan", "lainnya"
]

PAY_METHODS = [
    "cash", "bni", "bca", "mandiri", "bri", "bsi", "cimb", "danamon", "permata",
    "gopay", "ovo", "dana", "shopeepay",
    "cc_visa", "cc_mastercard", "cc_jcb", "cc_amex"
]

MERCHANTS = {
    "makanan":    ["Kopi Kenangan", "Warung Makan Bu Sri", "McD", "KFC", "Indomaret",
                   "Alfamart", "GoPizza", "Waroenk Steak", "Es Teh Pak Bagas", "Mie Ayam Pak Min"],
    "transport":  ["Pertamina", "Shell", "Gojek", "Grab", "Blue Bird", "KRL", "Transjakarta", "MRT Jakarta"],
    "belanja":    ["Shopee", "Tokopedia", "Lazada", "H&M", "Zara", "Uniqlo", "Matahari", "Hypermart"],
    "hiburan":    ["Netflix", "Spotify", "Cinema XXI", "CGV", "Steam", "YouTube Premium", "Disney+"],
    "kesehatan":  ["Apotek K24", "Guardian", "Halodoc", "RS Siloam", "Klinik Pratama", "Century Healthcare"],
    "pendidikan": ["Ruangguru", "Coursera", "Udemy", "Zenius", "SkillAcademy", "Dicoding"],
    "tagihan":    ["PLN", "Telkom", "PDAM", "Indihome", "XL", "Telkomsel", "Tri", "Smartfren"],
    "pemasukan":  ["Gaji Telkom", "Transfer BCA", "Freelance Toptal", "Dividen Saham", "Cashback OVO"],
    "lainnya":    ["ATM BCA", "Kantor Pos", "Notaris", "Laundry Bersih", "Bengkel Motor Pak Agus"],
}


def random_amount():
    """Return (amount_int, amount_text)."""
    base = random.choice([
        random.randint(5, 99) * 1000,
        random.randint(1, 9) * 10000,
        random.randint(1, 50) * 5000,
        random.randint(100, 500) * 1000,
    ])
    templates = [
        f"{base // 1000}rb",
        f"{base // 1000}ribu",
        f"{base // 1000}k",
        f"{base:,}".replace(",", "."),
        f"Rp{base // 1000}rb",
        f"Rp {base:,}".replace(",", "."),
    ]
    if base % 100000 == 0:
        templates.append(f"{base // 100000} ratus ribu")
    if base % 1000000 == 0:
        templates.append(f"{base // 1000000} juta")
    return base, random.choice(templates)


EXPENSE_TEMPLATES = [
    "{verb} {merchant} {amount} pakai {pay_method}",
    "{verb} {amount} di {merchant} pake {pay_method}",
    "{verb} {merchant} bayar {amount} via {pay_method}",
    "bayar {merchant} {amount} {pay_method}",
    "{verb} {amount} {merchant}",
    "{verb} di {merchant} {amount} {pay_method}",
    "{amount} buat {verb} di {merchant} pakai {pay_method}",
    "transaksi {merchant} {amount} metode {pay_method}",
]

INCOME_TEMPLATES = [
    "terima {amount} dari {merchant}",
    "masuk {amount} {pay_method} dari {merchant}",
    "transfer masuk {amount} {merchant}",
    "{merchant} kirim {amount} ke rekening",
    "gajian {amount} dari {merchant}",
    "dapat {amount} dari {merchant} via {pay_method}",
]

EXPENSE_VERBS = [
    "beli", "bayar", "jajan", "beli makan", "topup", "langganan",
    "isi bensin", "naik", "checkout", "order"
]


def generate_sample(category: str, transaction_type: str) -> dict:
    merchant = random.choice(MERCHANTS[category])
    amount_int, amount_text = random_amount()
    pay_method = random.choice(PAY_METHODS)

    if transaction_type == "expense":
        template = random.choice(EXPENSE_TEMPLATES)
        verb = random.choice(EXPENSE_VERBS)
        text = template.format(
            verb=verb, merchant=merchant,
            amount=amount_text, pay_method=pay_method
        )
    else:
        template = random.choice(INCOME_TEMPLATES)
        text = template.format(
            merchant=merchant, amount=amount_text, pay_method=pay_method
        )

    return {
        "text":        text,
        "amount":      amount_int,
        "amount_text": amount_text,
        "category":    category,
        "merchant":    merchant,
        "pay_method":  pay_method,
        "type":        transaction_type,
    }


def main(n: int, output_path: str):
    samples = []
    for _ in range(n):
        if random.random() < 0.8:
            tx_type  = "expense"
            category = random.choice([c for c in CATEGORY_LABELS if c != "pemasukan"])
        else:
            tx_type  = "income"
            category = "pemasukan"
        samples.append(generate_sample(category, tx_type))

    random.shuffle(samples)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[data_generation] {len(samples)} samples -> {output_path}")
    cat_dist = {}
    for s in samples:
        cat_dist[s["category"]] = cat_dist.get(s["category"], 0) + 1
    for k, v in sorted(cat_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic/transactions.json")
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    main(args.n, args.output)

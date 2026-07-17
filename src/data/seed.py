"""Builds data/debate.db: the same recommerce dataset (products, customers,
orders, order_items, returns; ~5k order line items) as the Data-Analyst-Agent
project, ported byte-for-byte in generation logic (random.seed(42)) so the
gold answers in eval/dataset.py -- carried over from that project's eval
set -- stay valid here too."""
import random
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "debate.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

PRODUCTS = [
    ("Wireless Earbuds", "Electronics", 59.99),
    ("Bluetooth Speaker", "Electronics", 89.99),
    ("Smart Watch", "Electronics", 199.99),
    ("Noise Cancelling Headphones", "Electronics", 249.99),
    ("Portable Charger", "Electronics", 34.99),
    ("Tablet Stand", "Electronics", 24.99),
    ("Running Shoes", "Sporting Goods", 79.99),
    ("Yoga Mat", "Sporting Goods", 29.99),
    ("Dumbbell Set", "Sporting Goods", 119.99),
    ("Camping Tent", "Sporting Goods", 159.99),
    ("Water Bottle", "Sporting Goods", 19.99),
    ("Winter Jacket", "Apparel", 129.99),
    ("Cotton T-Shirt", "Apparel", 19.99),
    ("Denim Jeans", "Apparel", 64.99),
    ("Wool Sweater", "Apparel", 74.99),
    ("Rain Jacket", "Apparel", 89.99),
    ("Coffee Maker", "Home", 69.99),
    ("Desk Lamp", "Home", 34.99),
    ("Blender", "Home", 54.99),
    ("Cookware Set", "Home", 149.99),
    ("Throw Blanket", "Home", 39.99),
    ("Facial Moisturizer", "Beauty", 22.99),
    ("Electric Toothbrush", "Beauty", 44.99),
    ("Hair Dryer", "Beauty", 39.99),
    ("Shampoo Set", "Beauty", 27.99),
]

FIRST_NAMES = [
    "Alice", "Ben", "Carla", "Dev", "Ella", "Farid", "Grace", "Hassan", "Ivy",
    "Jon", "Kira", "Liam", "Maya", "Noah", "Priya", "Quinn", "Rosa", "Sam",
    "Tara", "Umar", "Vera", "Wes", "Xena", "Yusuf", "Zoe",
]
LAST_NAMES = [
    "Chen", "Torres", "Diaz", "Patel", "Novak", "Khan", "Kim", "Ali",
    "Sullivan", "Meyer", "Volkov", "O'Brien", "Singh", "Reyes", "Nguyen",
    "Brooks", "Osei", "Farah", "Lindgren", "Petrov",
]

RETURN_REASONS = ["wrong_size", "defective", "not_as_described", "changed_mind", "wrong_item"]

MONTHS = [f"2025-{m:02d}" for m in range(1, 13)]

CUSTOMER_COUNT = 150
ORDERS_PER_MONTH = 235


def _build_customers():
    combos = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    return random.sample(combos, CUSTOMER_COUNT)


def build():
    random.seed(42)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())

    customers = _build_customers()

    conn.executemany(
        "INSERT INTO products (id, name, category, unit_price) VALUES (?, ?, ?, ?)",
        [(i + 1, n, c, p) for i, (n, c, p) in enumerate(PRODUCTS)],
    )
    conn.executemany(
        "INSERT INTO customers (id, name) VALUES (?, ?)",
        [(i + 1, n) for i, n in enumerate(customers)],
    )

    order_id = 1
    item_id = 1
    order_rows = []
    item_rows = []

    for month in MONTHS:
        for _ in range(ORDERS_PER_MONTH):
            customer_id = random.randint(1, CUSTOMER_COUNT)
            day = random.randint(1, 28)
            order_date = f"{month}-{day:02d}"
            order_rows.append((order_id, customer_id, order_date, "completed"))

            n_items = random.choice([1, 1, 2, 2, 3])
            chosen_products = random.sample(range(1, len(PRODUCTS) + 1), n_items)
            for product_id in chosen_products:
                quantity = random.choice([1, 1, 1, 2, 3])
                unit_price = PRODUCTS[product_id - 1][2]
                item_rows.append((item_id, order_id, product_id, quantity, unit_price))
                item_id += 1

            order_id += 1

    conn.executemany(
        "INSERT INTO orders (id, customer_id, order_date, status) VALUES (?, ?, ?, ?)",
        order_rows,
    )
    conn.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?)",
        item_rows,
    )

    order_date_by_id = {row[0]: row[2] for row in order_rows}

    return_rows = []
    return_id = 1
    for item in item_rows:
        if random.random() < 0.12:
            iid = item[0]
            order_date = order_date_by_id[item[1]]
            return_date = order_date[:7] + f"-{min(28, int(order_date[-2:]) + 5):02d}"
            reason = random.choice(RETURN_REASONS)
            return_rows.append((return_id, iid, reason, return_date))
            return_id += 1

    conn.executemany(
        "INSERT INTO returns (id, order_item_id, reason, return_date) VALUES (?, ?, ?, ?)",
        return_rows,
    )

    conn.commit()
    conn.close()
    print(f"Seeded {len(order_rows)} orders, {len(item_rows)} line items, {len(return_rows)} returns -> {DB_PATH}")


if __name__ == "__main__":
    build()

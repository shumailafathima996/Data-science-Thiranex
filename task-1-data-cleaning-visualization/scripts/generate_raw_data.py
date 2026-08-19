"""
Generates a realistic "messy" raw e-commerce sales dataset
to simulate real-world data quality issues for the cleaning project.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

n = 1000

products = ["Wireless Mouse", "Bluetooth Speaker", "Laptop Stand", "USB-C Hub",
            "Mechanical Keyboard", "Webcam HD", "Desk Lamp", "Phone Case",
            "Power Bank", "Noise Cancelling Headphones"]
categories = {
    "Wireless Mouse": "Accessories", "Bluetooth Speaker": "Audio",
    "Laptop Stand": "Office", "USB-C Hub": "Accessories",
    "Mechanical Keyboard": "Accessories", "Webcam HD": "Electronics",
    "Desk Lamp": "Office", "Phone Case": "Accessories",
    "Power Bank": "Electronics", "Noise Cancelling Headphones": "Audio"
}
regions = ["North", "South", "East", "West", " North", "south", "EAST", "West "]
payment_methods = ["Credit Card", "Debit Card", "UPI", "Cash", "credit card", "COD", np.nan]

rows = []
start_date = datetime(2025, 1, 1)

for i in range(n):
    product = random.choice(products)
    category = categories[product]
    base_price = {
        "Wireless Mouse": 15, "Bluetooth Speaker": 45, "Laptop Stand": 25,
        "USB-C Hub": 20, "Mechanical Keyboard": 60, "Webcam HD": 35,
        "Desk Lamp": 18, "Phone Case": 10, "Power Bank": 30,
        "Noise Cancelling Headphones": 90
    }[product]

    price = round(np.random.normal(base_price, base_price * 0.15), 2)
    if random.random() < 0.02:
        price = price * random.choice([15, -1, 0])

    quantity = np.random.poisson(3) + 1
    if random.random() < 0.015:
        quantity = random.choice([500, -5, 0])

    date = start_date + timedelta(days=random.randint(0, 240))
    date_str = date.strftime("%Y-%m-%d") if random.random() > 0.3 else date.strftime("%d/%m/%Y")

    customer_age = int(np.clip(np.random.normal(34, 10), 16, 75))
    if random.random() < 0.05:
        customer_age = np.nan

    region = random.choice(regions)
    payment = random.choice(payment_methods)

    rating = round(np.clip(np.random.normal(4.1, 0.9), 1, 5), 1)
    if random.random() < 0.08:
        rating = np.nan

    order_id = f"ORD{1000+i}"

    rows.append({
        "OrderID": order_id,
        "Order Date": date_str,
        "Product": product if random.random() > 0.03 else product.upper(),
        "Category": category,
        "Region": region,
        "Price": price if random.random() > 0.04 else np.nan,
        "Quantity": quantity,
        "CustomerAge": customer_age,
        "PaymentMethod": payment,
        "Rating": rating
    })

df = pd.DataFrame(rows)

dupes = df.sample(40, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

for _ in range(5):
    df.loc[len(df)] = [np.nan] * len(df.columns)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("data/raw_sales_data.csv", index=False)
print("Raw dataset created:", df.shape)
print(df.isna().sum())

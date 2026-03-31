from sklearn.datasets import fetch_openml
import pandas as pd

print("Querying real E-commerce data from OpenML. This might take a few seconds...")

# Query the Online Shoppers Intention dataset (Dataset ID: 42890)
# as_frame=True ensures we get a nice Pandas DataFrame
# Fetch the exact E-commerce dataset by name
shoppers_data = fetch_openml(name="online_shoppers_intention", version=1, as_frame=True, parser='auto')

df = shoppers_data.frame

# Save the queried data directly to your data folder
df.to_csv("data/dataset.csv", index=False)

print(f"Success! Queried and saved {len(df)} rows and {len(df.columns)} columns to data/dataset.csv.")
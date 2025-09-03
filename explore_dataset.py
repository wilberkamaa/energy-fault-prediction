import pandas as pd

# Load the dataset
df = pd.read_parquet('data/sample/hybrid_system_sample.parquet')

# Display information about the dataset
print("---\nDataset Head---")
print(df.head())
print("\n---\nDataset Shape---")
print(df.shape)
print("\n---\nDataset Columns---")
print(df.columns)
print("\n---\nDataset Info---")
df.info()
print("\n---\nDataset Description---")
print(df.describe())

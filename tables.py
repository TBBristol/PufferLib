import pandas as pd

url = "https://trackleaders.com/cocodona25"
tables = pd.read_html(url)

# This may give you multiple tables. Find the one with CPs.
for i, table in enumerate(tables):
    print(f"Table {i}:")
    print(table.head())

# Once you find the correct table:
checkpoint_table = tables[1]  # adjust index based on output

checkpoint_table.to_csv("cocodona_cp_times.csv", index=False)

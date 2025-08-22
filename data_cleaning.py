# scripts/data_cleaning.py

import pandas as pd
import numpy as np
import os

# Folder Paths
dataset_folder = "datasets/"
cleaned_folder = "cleaned_data/"
report_file = os.path.join(cleaned_folder, "data_cleaning_report.html")

os.makedirs(cleaned_folder, exist_ok=True)

# Load datasets
dataset1 = pd.read_csv(os.path.join(dataset_folder, "dataset1.csv"))
dataset2 = pd.read_csv(os.path.join(dataset_folder, "dataset2.csv"))

# Keep raw copies for "before vs after" summary
raw1 = dataset1.copy()
raw2 = dataset2.copy()

# Explore data (printed to terminal)
def explore_data(df, name):
    print(f"--- {name} Info ---")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    print("\n---------------------------------\n")

explore_data(dataset1, "Dataset 1")
explore_data(dataset2, "Dataset 2")

# Handle Missing Data
def handle_missing(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include="object").columns
    # Fill numeric with mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # Fill categorical with mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

dataset1 = handle_missing(dataset1)
dataset2 = handle_missing(dataset2)

# Remove duplicates
dup1_before = raw1.duplicated().sum()
dup2_before = raw2.duplicated().sum()
dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)

# Standardize column names
dataset1.columns = dataset1.columns.str.lower().str.replace(" ", "_", regex=False)
dataset2.columns = dataset2.columns.str.lower().str.replace(" ", "_", regex=False)

# Outlier Removal (IQR) for numeric columns
def remove_outliers(df):
    clean_df = df.copy()
    numeric_cols = clean_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]
    return clean_df

dataset1_clean = remove_outliers(dataset1)
dataset2_clean = remove_outliers(dataset2)

# Save cleaned data (CSV + HTML tables)
dataset1_clean.to_csv(os.path.join(cleaned_folder, "dataset1_clean.csv"), index=False)
dataset2_clean.to_csv(os.path.join(cleaned_folder, "dataset2_clean.csv"), index=False)
dataset1_clean.to_html(os.path.join(cleaned_folder, "dataset1_clean.html"), index=False)
dataset2_clean.to_html(os.path.join(cleaned_folder, "dataset2_clean.html"), index=False)

# Build summary tables for the report
def profile(df):
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": ", ".join(df.select_dtypes(include=np.number).columns.tolist()) or "(none)"
    }

summary1_before = pd.DataFrame([profile(raw1)])
summary1_after  = pd.DataFrame([profile(dataset1_clean)])
summary2_before = pd.DataFrame([profile(raw2)])
summary2_after  = pd.DataFrame([profile(dataset2_clean)])

# Generate HTML report (tables only—no plots)
html_content = f"""
<html>
<head>
    <meta charset="utf-8"/>
    <title>Data Cleaning Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; }}
        h1 {{ margin-bottom: 6px; }}
        .muted {{ color: #666; margin-top: 0; }}
        h2 {{ color: #0d6efd; margin-top: 28px; }}
        table {{ border-collapse: collapse; margin: 12px 0 28px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 14px; }}
        th {{ background: #f6f8fa; }}
        code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
        .grid {{ display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
        .card {{ border: 1px solid #eee; padding: 12px 16px; border-radius: 10px; }}
        .small {{ font-size: 13px; color: #555; }}
    </style>
</head>
<body>
    <h1>Data Cleaning Project Report</h1>
    <p class="muted">Covers: Missing data handling, duplicates removal, standardization, outlier detection & cleaned outputs.</p>

    <h2>Dataset 1 — Summary</h2>
    <div class="grid">
        <div class="card">
            <h3>Before Cleaning</h3>
            {summary1_before.to_html(index=False)}
        </div>
        <div class="card">
            <h3>After Cleaning</h3>
            {summary1_after.to_html(index=False)}
        </div>
    </div>

    <h3>Dataset 1 — First 50 Cleaned Rows</h3>
    {dataset1_clean.head(50).to_html(index=False)}

    <h2>Dataset 2 — Summary</h2>
    <div class="grid">
        <div class="card">
            <h3>Before Cleaning</h3>
            {summary2_before.to_html(index=False)}
        </div>
        <div class="card">
            <h3>After Cleaning</h3>
            {summary2_after.to_html(index=False)}
        </div>
    </div>

    <h3>Dataset 2 — First 50 Cleaned Rows</h3>
    {dataset2_clean.head(50).to_html(index=False)}

    <p class="small"><strong>Notes:</strong> Numeric missing values imputed with mean; categorical with mode. Duplicates dropped. Column names standardized to lowercase with underscores. Outliers removed via IQR rule per numeric column.</p>
</body>
</html>
"""

with open(report_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print("Data cleaning complete!")
print(f"Cleaned CSVs & HTML pages saved in '{cleaned_folder}'.")
print(f"Full HTML report saved as '{report_file}'. Open it in a browser to view.")

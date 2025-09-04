import pandas as pd
import os
import config

def generate_data_quality_report():
    print("--- Generating Data Quality Report for the Final Dataset ---")

    file_path = config.PREPROCESSED_DATA_PATH

    if not os.path.exists(file_path):
        print(f"ERROR: Processed file not found at '{file_path}'")
        print("Please run 'run_preprocessing.py' first to generate the dataset.")
        return

    print(f"INFO: Reading data from: {file_path}\n")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        return

    print("--- 1. Basic Information ---")
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print("-" * 50)

    print("\n--- 2. Data Preview (First 5 Rows) ---")
    print(df.head())
    print("-" * 50)

    print("\n--- 3. Column Info & Data Types ---")
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())
    print("-" * 50)


    print("\n--- 4. Descriptive Statistics for Numerical Columns ---")
    print(df.describe().round(4))
    print("-" * 50)

    print("\n--- 5. Missing Value (-1) Count per Column ---")
    missing_values = df.isin([-1]).sum()
    missing_values_with_counts = missing_values[missing_values > 0]
    
    if missing_values_with_counts.empty:
        pass
    else:
        print("Columns with '-1' values found:")
        print(missing_values_with_counts.sort_values(ascending=False))
    print("-" * 50)


if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    generate_data_quality_report()
import os
import pandas as pd
import csv

def count_csv_rows_in_folder(folder_path):
    total_rows = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                # 读取 CSV 文件
                length = sum(1 for row in open(file_path,"r",encoding="utf-8"))


                # 累加非空行数
                total_rows += length
                print(f"File: {filename} - Rows: {length}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    return total_rows

if __name__ == "__main__":
    folder_path = r"C:\Users\Joe\Desktop\data\test"
    total_rows = count_csv_rows_in_folder(folder_path)
    print(f"Total rows in all CSV files (excluding blank rows): {total_rows}")
import os
import pandas as pd
import time
from datetime import datetime

def load_data(csv_path):
    return pd.read_csv(csv_path)

def save_batch_output(results, output_dir, batch_index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/batch_{batch_index}_{timestamp}.csv", index=False)
    print("File saved to: ",f"{output_dir}/batch_{batch_index}_{timestamp}.csv")

def log_time(output_dir, batch_index, duration, prefix):
    with open(f"{output_dir}/timing_log.txt", "a") as f:
        f.write(f"{prefix}_Batch_{batch_index}: {duration:.2f} seconds\n")

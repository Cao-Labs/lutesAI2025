import pandas as pd
import matplotlib.pyplot as plt
import os

# === Hardcoded Settings ===
csv_subdir = '2025-07-03_02-49'  # Subdirectory inside saved_models
csv_filename = 'training_log_test.csv'  # CSV file name
x_col = 'episode'
y_col = 'reward'
num_points = 100  # How many points to plot; set to None for all
title = 'Closer Look at Training Reward'
output_image = 'closerlook.png'  # Output image file name

# === Construct full path to CSV ===
csv_path = os.path.join("saved_models", csv_subdir, csv_filename)

# === Check if file exists ===
if not os.path.exists(csv_path):
    print(f"[ERROR] CSV file not found at: {csv_path}")
else:
    # Load CSV
    df = pd.read_csv(csv_path)

    # Check columns exist
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[ERROR] Columns '{x_col}' or '{y_col}' not found in CSV.")
    else:
        # Optional downsampling
        if num_points is not None and num_points < len(df):
            step = max(1, len(df) // num_points)
            df = df.iloc[::step]

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', alpha=0.8)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_image)
        print(f"[INFO] Plot saved to: {output_image}")
        plt.close()

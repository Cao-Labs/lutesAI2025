import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv_graph(csv_subdir, x_col='episode', y_col='reward', num_points=None, title="Graph", save_path=None):
    """
    Plots a graph from a CSV file with optional point sampling.

    Parameters:
        csv_subdir (str): Subdirectory under 'saved_models' where the CSV is stored.
        x_col (str): Column to use as x-axis (default: 'episode').
        y_col (str): Column to use as y-axis (default: 'reward').
        num_points (int, optional): Number of points to plot. If None, plot all.
        title (str): Title of the graph.
        save_path (str, optional): If provided, saves the graph to this path.
    """
    # Build full CSV path
    full_path = os.path.join("saved_models", csv_subdir, "training_log_test.csv")

    if not os.path.exists(full_path):
        print(f"[ERROR] CSV file not found at: {full_path}")
        return

    # Load data
    df = pd.read_csv(full_path)

    # Validate columns
    if x_col not in df.columns or y_col not in df.columns:
        print(f"[ERROR] Columns '{x_col}' or '{y_col}' not found in CSV.")
        return

    # Optional downsampling
    if num_points is not None and num_points < len(df):
        df = df.iloc[::len(df) // num_points]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', alpha=0.8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

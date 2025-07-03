import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_csv_graph(csv_path, x_col, y_col, num_points=None, title="Graph", save_path=None):
    """
    Plots a graph from a CSV file with optional point sampling.

    Parameters:
        csv_path (str): Path to the CSV file.
        x_col (str): Name of the column to use as the x-axis.
        y_col (str): Name of the column to use as the y-axis.
        num_points (int, optional): Number of points to plot. If None, plot all.
        title (str): Title of the graph.
        save_path (str, optional): If provided, saves the graph to this path.
    """

    # Load CSV
    print("what directory")
    path = input()
    print("How many points to plot?")
    num_points = int(input())

    # Correct way to build the path
    full_path = os.path.join("saved_models", csv_path, "training_log_test.csv")
    df = pd.read_csv(full_path)

    # Sample points if num_points is specified and less than total
    if num_points is not None and num_points < len(df):
        df = df.iloc[::len(df)//num_points]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], marker='o', linestyle='-', alpha=0.8)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

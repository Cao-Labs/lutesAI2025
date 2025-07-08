import pandas as pd
import matplotlib.pyplot as plt
import os

def graphit(Training_Path, Eval_Path, Graph_Path):
    # Load training data
    df = pd.read_csv(Training_Path, encoding='utf-8-sig')
    print("Training log columns:", df.columns.tolist())

    # Create directory if it doesn't exist
    os.makedirs(Graph_Path, exist_ok=True)
    N = max(1, int(df['episode'].max() / 500))
    df_sampled = df[df['episode'] % N == 0]

    # 1. Plot: Training reward + selected_amount
    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.plot(df_sampled['episode'], df_sampled['selected_amount'], label='Selected', alpha=0.7)

    # Sample every 10,000th point for overlay
    plt.xlabel('Episode')
    plt.ylabel('Number of Answers')
    plt.title('Selected Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Graph_Path, "selected.png"))
    plt.close()

    # 2. Plot: Training reward only
    plt.figure(figsize=(12, 6), constrained_layout=True)

    plt.plot(df_sampled['episode'], df_sampled['reward'], label='F1/Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('F1/Reward')
    plt.title('Training Reward')
    plt.grid(True)
    plt.savefig(os.path.join(Graph_Path, "training.png"))
    plt.close()

    # Recall and Precision
    plt.figure(figsize=(12, 6), constrained_layout=True)
    plt.plot(df_sampled['episode'], df_sampled['reward'], label='F1/Reward', alpha=0.7, color = "blue")
    plt.plot(df_sampled['episode'], df_sampled['precision'], label='precision', alpha=0.7, color = "red")
    plt.plot(df_sampled['episode'], df_sampled['recall'], label='recall', alpha=0.7, color = "green")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("F1/Precision/Recall Over Training")
    plt.grid(True)
    plt.savefig(os.path.join(Graph_Path, "overview.png"))
    plt.close()

    # 3. Plot: Evaluation data
    ef = pd.read_csv(Eval_Path, encoding='utf-8-sig')
    plt.figure(figsize=(10, 5))
    plt.plot(ef['episode'], ef['avg_reward'], label='Evaluation Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Evaluation Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_Path, "eval.png"))
    plt.close()

    #average correct
    ef = pd.read_csv(Eval_Path, encoding='utf-8-sig')
    plt.figure(figsize=(10, 5))
    plt.plot(ef['episode'], ef['avg_percent_correct'], label='Evaluation Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Correct')
    plt.title('average eval percent correct')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_Path, "eval_correct.png"))
    plt.close()

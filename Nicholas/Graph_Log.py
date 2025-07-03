import pandas as pd
import matplotlib.pyplot as plt
import os

def graphit(Training_Path, Eval_Path, Graph_Path):
    # Load training data
    df = pd.read_csv(Training_Path)

    # Create directory if it doesn't exist
    os.makedirs(Graph_Path, exist_ok=True)

    # 1. Plot: Training reward + selected_amount
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], label='Reward', alpha=0.7)

    # Sample every 10,000th point for overlay
    df_sampled = df[df['episode'] % 10000 == 0]
    plt.plot(df_sampled['episode'], df_sampled['selected_amount'], label='Number of Answers',
             linestyle='--', color='red', marker='o')

    plt.xlabel('Episode')
    plt.ylabel('Reward / Number of Answers')
    plt.title('Training Reward and Selected Amount')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_Path, "reward_selected.png"))
    plt.close()

    # 2. Plot: Training reward only
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], label='Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_Path, "training.png"))
    plt.close()

    # 3. Plot: Evaluation data
    ef = pd.read_csv(Eval_Path)
    plt.figure(figsize=(10, 5))
    plt.plot(ef['avg_reward'], ef['reward'], label='Evaluation Reward', alpha=0.7)
    plt.xlabel('Average Reward')
    plt.ylabel('Reward')
    plt.title('Evaluation Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Graph_Path, "eval.png"))
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt

# Load data

def graphit(Training_Path, Eval_Path,Graph_Path):

    df = pd.read_csv(Training_Path)

    # First: base plot with full training data
    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['reward'], label='reward', alpha=0.7)

    # Now: overlay eval data, but only every 20,000th point
    df_sampled = df[df['episode'] % 10000 == 0]

    # Overlay sampled evaluation data as a dotted red line or points
    plt.plot( df_sampled['episode'], df_sampled['selected_amount'], label='Number of answers', linestyle='--', color='red', marker='o')

    # Labels and legend
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training reward and chosen')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show plot
    plt.savefig(Graph_Path,"reward+selected")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot( df['episode'], df['reward'], label='reward', alpha=0.7)
    plt.savefig(Graph_Path,"training")
    plt.close()

    ef = pd.read_csv(Eval_Path)
    plt.figure(figsize=(10, 5))
    plt.plot(ef['avg_reward'], ef['reward'], label='reward', alpha=0.7)

    plt.savefig(Graph_Path,"eval")
    plt.close()






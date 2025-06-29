import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('training_log.csv')
ef = pd.read_csv('eval_log.csv')

# First: base plot with full training data
plt.figure(figsize=(10, 5))
plt.plot(df['episode'], df['reward'], label='Training', alpha=0.7)

# Now: overlay eval data, but only every 20,000th point
df_sampled = df[df['episode'] % 20000 == 0]

# Overlay sampled evaluation data as a dotted red line or points
plt.plot(df['episode'], df['num_selected_terms'], label='Eval (every 20k)', linestyle='--', color='red', marker='o')

# Labels and legend
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training reward and chosen')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
plt.savefig("combined")

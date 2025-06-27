import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("training_log.csv")

# Filter every 5000th episode
df_filtered = df[df['episode'] % 5000 == 0]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df_filtered['episode'], df_filtered['reward'], marker='o')

# Set x-axis ticks every 5000
plt.xticks(
    ticks=range(df_filtered['episode'].min(), df_filtered['episode'].max() + 5000, 5000)
)

# Optional: Zoom in to a specific range, e.g., from 5000 to 25000
# Adjust these limits as needed
plt.xlim(5000, 25000)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Every 5000 Episodes (Zoomed In)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(" every 5000 Episodes.png")
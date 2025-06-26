# torch_rl_go_agent.py

import torch
import csv
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time
from go_env import GOEnv, proteins, all_go_terms, protein_data_for_env  # You wrote this already

startTime = time.time()
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_DIR = os.path.join(SAVE_DIR, "best")
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
# Create folders if they don't exist
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------- 1. Define the PyTorch Policy Network ---------
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, output_size=100):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Embedding(22, 32),           # Encode 22 amino acids to vectors
            nn.Flatten(),                   # (512, 32) → (512*32)
            nn.Linear(512 * 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Since output is a binary vector
        )
        baseline = 0.0

    def forward(self, x):
        return self.model(x)

# --------- 2. Train with REINFORCE ---------
def train(env, policy, optimizer, episodes=500, eval_log ='eval_log.csv', training_log = 'training_log.csv' ):
    baseline = 0.0
    best_avg_reward = -float('inf')

    # Create CSV file and write header (if not already exists)
    if not os.path.exists(eval_log):
        with open(eval_log, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward','time'])  # header
    if not os.path.exists(training_log):
        with open(training_log, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'time'])
    for episode in range(episodes):
        obs = env.reset()
        sequence = torch.tensor(obs["sequence"], dtype=torch.long).unsqueeze(0)
        probs = policy(sequence).squeeze(0)

        dist = torch.distributions.Bernoulli(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)

        action_np = action.detach().numpy().astype(np.int8)
        _, reward, _, _ = env.step(action_np)

        # REINFORCE loss
        baseline = 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        loss = -advantage * log_probs.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with open(training_log, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, time.time()])

        if episode % 100 == 0:
            print(f"\n🎯 Episode {episode}: Reward = {reward:.2f}, Baseline = {baseline:.2f}")
            avg_reward, best_avg_reward = evaluate(policy, env, episodes=10, episode_idx=episode, best_avg_reward=best_avg_reward)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_episode_{episode}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"📝 Checkpoint saved to {checkpoint_path}")

            # Append to CSV log
            with open(eval_log, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, avg_reward,time.time()])

def evaluate(policy, env, episodes=20, episode_idx=0, best_avg_reward=None):
    total_reward = 0

    for _ in range(episodes):
        obs = env.reset()
        sequence = torch.tensor(obs["sequence"], dtype=torch.long).unsqueeze(0)
        probs = policy(sequence).squeeze(0)

        action = (probs > 0.5).int().numpy().astype(np.int8)
        _, reward, _, _ = env.step(action)
        total_reward += reward

    avg_reward = total_reward / episodes
    print(f"✅ Evaluation at Episode {episode_idx}: Avg Reward = {avg_reward:.2f}")

    # Save best model
    if best_avg_reward is None or avg_reward > best_avg_reward:
        best_path = os.path.join(BEST_MODEL_DIR, f"best_model_episode_{episode_idx}.pt")
        torch.save(policy.state_dict(), best_path)
        print(f"💾 Best model saved to {best_path}")
        best_avg_reward = avg_reward  # ✅ Update best reward

    return avg_reward, best_avg_reward  # ✅ Always return both

# --------- 3. Run Everything ---------
if __name__ == "__main__":
    env = GOEnv(protein_data=protein_data_for_env, go_terms=all_go_terms)

    policy = PolicyNetwork(input_size=512, output_size=env.max_choices)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    train(env, policy, optimizer, episodes=101)

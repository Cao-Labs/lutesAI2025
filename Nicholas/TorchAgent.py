# torch_rl_go_agent.py

import torch
import csv
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from go_env import GOEnv, proteins, all_go_terms, protein_data_for_env  # You wrote this already

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
def train(env, policy, optimizer, episodes=500, log_file='reward_log.csv'):
    baseline = 0.0
    best_avg_reward = -float('inf')

    # Create CSV file and write header (if not already exists)
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward'])  # header

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

        if episode % 50 == 0:
            print(f"\n🎯 Episode {episode}: Reward = {reward:.2f}, Baseline = {baseline:.2f}")
            avg_reward = evaluate(policy, env, episodes=10, episode_idx=episode, best_avg_reward=best_avg_reward)

            # Update best model if needed
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward

            # Append to CSV log
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, avg_reward])

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

    # Save model if best
    if best_avg_reward is None or avg_reward > best_avg_reward:
        torch.save(policy.state_dict(), f'best_model_episode_{episode_idx}.pt')
        print(f"💾 Model saved at episode {episode_idx} with avg reward {avg_reward:.2f}")
        return avg_reward

    return best_avg_reward

# --------- 3. Run Everything ---------
if __name__ == "__main__":
    env = GOEnv(protein_data=protein_data_for_env, go_terms=all_go_terms)

    policy = PolicyNetwork(input_size=512, output_size=env.max_choices)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    train(env, policy, optimizer, episodes=100)

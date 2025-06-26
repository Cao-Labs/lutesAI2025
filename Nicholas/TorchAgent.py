# torch_rl_go_agent.py

import torch
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
def train(env, policy, optimizer, episodes=500):
    baseline = 0.0
    for episode in range(episodes):
        obs = env.reset()
        sequence = torch.tensor(obs["sequence"], dtype=torch.long).unsqueeze(0)  # Shape (1, 512)
        go_candidates = obs["go_candidates"]

        probs = policy(sequence).squeeze(0)  # Shape (100,)
        dist = torch.distributions.Bernoulli(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)

        # Convert action from Tensor to NumPy for env.step
        action_np = action.detach().numpy().astype(np.int8)
        _, reward, _, _ = env.step(action_np)

        # REINFORCE loss: maximize reward-weighted log prob
        baseline = 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        loss = -advantage * log_probs.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {reward}")
            evaluate(env, policy, episode)
def evaluate(policy, env, episodes=20):
    total_reward = 0

    for _ in range(episodes):
        obs = env.reset()
        sequence = torch.tensor(obs["sequence"], dtype=torch.long).unsqueeze(0)
        probs = policy(sequence).squeeze(0)

        # Select GO terms with probability > 0.5
        action = (probs > 0.5).int().numpy().astype(np.int8)

        _, reward, _, _ = env.step(action)
        total_reward += reward

    avg_reward = total_reward / episodes
    print(f"Avg Reward over {episodes} episodes: {avg_reward:.2f}")

# --------- 3. Run Everything ---------
if __name__ == "__main__":
    env = GOEnv(protein_data=protein_data_for_env, go_terms=all_go_terms)

    policy = PolicyNetwork(input_size=512, output_size=env.max_choices)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    train(env, policy, optimizer, episodes=100)

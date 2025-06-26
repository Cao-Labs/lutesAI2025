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

    def forward(self, x):
        return self.model(x)

# --------- 2. Train with REINFORCE ---------
def train(env, policy, optimizer, episodes=500):
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
        loss = -log_probs.sum() * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {reward}")

# --------- 3. Run Everything ---------
if __name__ == "__main__":
    env = GOEnv(protein_data=protein_data_for_env, go_terms=all_go_terms)

    policy = PolicyNetwork(input_size=512, output_size=env.max_choices)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    train(env, policy, optimizer, episodes=1000)

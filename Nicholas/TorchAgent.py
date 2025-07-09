# torch_rl_go_agent.py

import torch
import csv
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import time
import datetime
from go_env import GOEnv, proteins, all_go_terms, protein_data_for_env # You wrote this already
from Graph_Log import graphit
whenRan = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR,whenRan), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR,whenRan, "graph"), exist_ok=True)
BEST_MODEL_DIR = os.path.join(SAVE_DIR,whenRan, "best")
CHECKPOINT_DIR = os.path.join(SAVE_DIR,whenRan, "checkpoints")
# Create folders if they don't exist
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # def to_device(tensor, device):
    #     return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor

# --------- 2. Train with REINFORCE ---------
def train(env, policy, optimizer, episodes=500, eval_log='eval_log.csv', training_log='training_log_test.csv'):
    startTime = time.time()
    baseline = 0.0
    best_avg_reward = -float('inf')
    Reset_Interval = 25000
    # Running mean and std for reward normalization
    reward_sum = 0.0
    reward_sq_sum = 0.0
    reward_count = 0
    epsilon = 1e-8  # to avoid division by zero

    # Create CSV headers
    training_log_path = os.path.join(SAVE_DIR, whenRan, training_log)
    eval_log_path = os.path.join(SAVE_DIR, whenRan, eval_log)

    if not os.path.exists(eval_log_path):
        with open(eval_log_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward', 'avg_percent_correct', 'time'])

    if not os.path.exists(training_log_path):
        with open(training_log_path, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'selected_amount', 'percent_correct','precision','recall','selected_go' 'time'])

    for episode in range(episodes):
        obs = env.reset()
        sequence = torch.tensor(obs["sequence"], dtype=torch.long, device=device).unsqueeze(0)
        probs = policy(sequence).squeeze(0)

        dist = torch.distributions.Bernoulli(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)

        action_np = action.detach().cpu().numpy().astype(np.int8)  # send back to CPU for env
        num_selected_terms = int(np.sum(action_np))  # Total GO terms selected
        _, reward, _, info = env.step(action_np)
        percent_correct = info.get("percent_correct", 0.0)
        precision = info.get("precision", 0.0)
        recall = info.get("recall", 0.0)
        select = info.get("selected_go", 0.0)
        selected_go = ",".join(select)

        # Update running stats useless for now
        # reward_sum += reward
        # reward_sq_sum += reward ** 2
        # reward_count += 1
        #
        # running_mean = reward_sum / reward_count
        # running_var = (reward_sq_sum / reward_count) - (running_mean ** 2)
        # running_std = running_var ** 0.5
        #
        # # Normalize reward
        # normalized_reward = (reward - running_mean) / (running_std + epsilon)
        #
        # # REINFORCE loss using normalized reward advantage
        # baseline = 0.9 * baseline + 0.1 * normalized_reward  # baseline updated with normalized reward
        advantage = reward
        loss = -advantage * log_probs.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Log Episode
        with open(os.path.join(SAVE_DIR,whenRan,training_log), mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, num_selected_terms,percent_correct,precision,recall,selected_go,time.time() - startTime])
        # if episode > 0 and episode % Reset_Interval == 0:
        #     reward_sum = 0.0
        #     reward_sq_sum = 0.0
        #     reward_count = 0
        #     print(f"🔄 Reset running stats at episode {episode}")

        if episode % 250 == 0:
            print(
                f"\n🎯 Episode {episode}: Reward = {reward:.2f}, Baseline = {baseline:.3f}")
            avg_reward, best_avg_reward, avg_percent = evaluate(policy, env, episodes=10, episode_idx=episode,
                                                   best_avg_reward=best_avg_reward)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_episode_{episode}.pt")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"📝 Checkpoint saved to {checkpoint_path}")

            with open(os.path.join(SAVE_DIR,whenRan,eval_log), mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, avg_reward, avg_percent, time.time() - startTime])

def evaluate(policy, env, episodes=20, episode_idx=0, best_avg_reward=None):
    policy.eval()
    total_reward = 0
    total_percent = 0
    with torch.no_grad():
        for _ in range(episodes):
            obs = env.reset()
            sequence = torch.tensor(obs["sequence"], dtype=torch.long, device=device).unsqueeze(0)
            probs = policy(sequence).squeeze(0)

            action = (probs > 0.5).int().cpu().numpy().astype(np.int8)
            _, reward, _, info = env.step(action)
            percent_correct = info.get("percent_correct", 0.0)
            total_percent += percent_correct
            total_reward += reward

    avg_percent = total_percent / episodes
    avg_reward = total_reward / episodes
    print(f"✅ Evaluation at Episode {episode_idx}: Avg Reward = {avg_reward:.2f}")

    # Save best model
    if best_avg_reward is None or avg_reward > best_avg_reward:
        path = os.path.join(os.path.join(BEST_MODEL_DIR), f"best_model_episode_{episode_idx}.pt")
        torch.save(policy.state_dict(), path)
        print(f"💾 Saved best model to {path}")
        best_avg_reward = avg_reward

    policy.train()

    return avg_reward, best_avg_reward,avg_percent  # ✅ Always return both

# --------- 3. Run Everything ---------
if __name__ == "__main__":
    env = GOEnv(protein_data=protein_data_for_env, go_terms=all_go_terms)
    policy = PolicyNetwork(input_size=512, output_size=env.max_choices).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    train(env, policy, optimizer, episodes=50000)

    graphit(os.path.join(SAVE_DIR,whenRan,'training_log_test.csv'), os.path.join(SAVE_DIR,whenRan,'eval_log.csv'),os.path.join(SAVE_DIR,whenRan,"graph"))

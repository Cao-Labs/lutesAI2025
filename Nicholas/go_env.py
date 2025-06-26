import random
import re
import gym
import numpy as np
import random
from gym import spaces
file_path = "../../../databases/UniProt2025/training_data_processedUniprot_DB.txt"  # Change if needed
proteins = {}

current_id = None
current_go = None
sequence_lines = []

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # New protein entry starts with an ID like "027R_FRG3G"
        if re.match(r'^\w+_\w+', line):
            if current_id:
                proteins[current_id] = {
                    "go_terms": current_go.split(';'),
                    "sequence": ''.join(sequence_lines)
                }

            parts = re.split(r'\s+', line)
            current_id = parts[0]
            current_go = parts[2]
            sequence_lines = [''.join(parts[3:])]  # Remainder of line is part of sequence
        else:
            # Continuation of sequence on a new line
            sequence_lines.append(line)

    # Save the last entry
    if current_id:
        proteins[current_id] = {
            "go_terms": current_go.split(';'),
            "sequence": ''.join(sequence_lines)
        }

# ✅ Print example
proteinSpecific = random.choice(list(proteins.items()))
protein_id, info = proteinSpecific

# extract GO terms
go_terms = info['go_terms']
sequence = info['sequence']
print(go_terms)
print(sequence)




class GOEnv(gym.Env):
    def __init__(self, protein_data, go_terms, max_choices=100, seq_length=512):
        super(GOEnv, self).__init__()
        self.protein_data = protein_data  # {protein_id: (sequence, [GO terms])}
        self.go_terms = go_terms  # list of all possible GO terms
        self.max_choices = max_choices
        self.seq_length = seq_length

        # Observation: encoded protein sequence
        self.observation_space = spaces.Box(low=0, high=22, shape=(seq_length,), dtype=np.int32)

        # Action: binary vector of GO term guesses (0 or 1 for each of 100 terms)
        self.action_space = spaces.MultiBinary(max_choices)

        self.current_sequence = None
        self.current_go_choices = []
        self.current_truth = []

    def encode_sequence(self, sequence):
        # Simplified amino acid to integer encoding
        aa_map = {aa: idx for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        encoded = [aa_map.get(aa, 20) for aa in sequence[:self.seq_length]]
        unknown_token = 20
        pad_token = 21

        encoded = [aa_map.get(aa, unknown_token) for aa in sequence[:self.seq_length]]
        return np.pad(encoded, (0, self.seq_length - len(encoded)), constant_values=pad_token)

    def reset(self):
        protein_id, (seq, true_go_terms) = random.choice(list(self.protein_data.items()))
        self.current_sequence = self.encode_sequence(seq)
        self.current_truth = true_go_terms
        # Ensure true GO terms are unique
        true_terms = set(true_go_terms)
        # Sample distractors from GO terms that are NOT in the true set
        possible_distractors = list(set(self.go_terms) - true_terms)
        # Calculate how many more terms we need
        num_distractors = self.max_choices - len(true_terms)
        # Sample without replacement to avoid duplicates
        distractors = random.sample(possible_distractors, min(num_distractors, len(possible_distractors)))
        # Combine and shuffle
        self.current_go_choices = list(true_terms.union(distractors))
        random.shuffle(self.current_go_choices)

        return {
            "sequence": self.current_sequence,
            "go_candidates": self.current_go_choices
        }

    def step(self, action):
        selected_terms = [self.current_go_choices[i] for i in range(self.max_choices) if action[i] == 1]
        correct = set(selected_terms) & set(self.current_truth)
        incorrect = set(selected_terms) - correct
        incorrectScore = len(incorrect) *.25
        reward = len(correct)- (incorrectScore)
        # Calculate percentage of correct GO terms guessed
        total_true = len(set(self.current_truth))
        percent_correct = (len(correct) / total_true) * 100 if total_true > 0 else 0.0

        # Optional: log info
        # print(f"Reward: {reward} / {total_true} true GO terms")
        # print(f"Percent correct: {percent_correct:.2f}%")
        # print(f"Selected terms: {selected_terms}")
        # print(f"Correct terms: {correct}")

        done = True

        return {
            "sequence": self.current_sequence,
            "go_candidates": self.current_go_choices
        }, reward, done, {}


protein_data_for_env = {
    pid: (info["sequence"], info["go_terms"])
    for pid, info in proteins.items()
}

all_go_terms = list({go for info in proteins.values() for go in info["go_terms"]})

env = GOEnv(protein_data=protein_data_for_env, go_terms=all_go_terms)

# Now you can use env.reset(), env.step() in your RL loop
obs = env.reset()
#print("Sequence shape:", obs["sequence"].shape)
#print("GO candidates:", obs["go_candidates"][:10])  # first 10 GO terms

action = np.zeros(env.max_choices, dtype=np.int8)
action[:5] = 1  # pretend agent selects first 5 GO terms

next_obs, reward, done, _ = env.step(action)
# print("Reward:", reward)# for prot_id, info in proteins.items():
#     print(f"Name: {prot_id}")
#     print(f"GO terms: {info['go_terms']}")
#     print(f"Sequence: {info['sequence'][:60]}...")  # Truncated for readability
#     print("-" * 40)

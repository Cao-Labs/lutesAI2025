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
hundred_over = 0
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
    # ✅ Filter out proteins with more than 100 GO terms
    original_count = len(proteins)
    proteins = {
        pid: info for pid, info in proteins.items()
        if len(info["go_terms"]) <= 100
    }
    filtered_count = len(proteins)
    print(
        f"[INFO] Filtered proteins: kept {filtered_count} of {original_count} (removed {original_count - filtered_count}) entries with ≤ 100 GO terms")
# ✅ Print example
proteinSpecific = random.choice(list(proteins.items()))
protein_id, info = proteinSpecific

# extract GO terms
go_terms = info['go_terms']
sequence = info['sequence']
print(go_terms)
print(sequence)




class GOEnv(gym.Env):
    def __init__(self, protein_data, go_terms, max_choices=100, seq_length=1300):
        super(GOEnv, self).__init__()
        self.protein_data = protein_data  # {protein_id: (sequence, [GO terms])}
        self.go_terms = go_terms  # list of all possible GO terms
        self.max_choices = max_choices
        self.seq_length = seq_length
        self.protein_ids = list(self.protein_data.keys())  # Precompute once
        self.go_term_set = set(self.go_terms)  # Precompute once

        self.encoded_proteins = {
            pid: (self.encode_sequence(seq), go_terms)
            for pid, (seq, go_terms) in self.protein_data.items()
        }
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
        # encoded = [aa_map.get(aa, 20) for aa in sequence[:self.seq_length]]
        unknown_token = 20
        pad_token = 21

        encoded = [aa_map.get(aa, unknown_token) for aa in sequence[:self.seq_length]]
        return np.pad(encoded, (0, self.seq_length - len(encoded)), constant_values=pad_token)

    def reset(self):
        # Choose a random protein ID from the precomputed list
        protein_id = random.choice(self.protein_ids)


        # Encode sequence and store ground truth
        self.current_sequence, true_go_terms = self.encoded_proteins[protein_id]
        self.current_truth = true_go_terms

        # Ensure true GO terms are unique
        true_terms = set(true_go_terms)
        # Use precomputed set of all GO terms (set(self.go_terms)) -> stored as self.go_term_set
        possible_distractors = list(self.go_term_set - true_terms)
        # Calculate how many more terms we need
        num_distractors = self.max_choices - len(true_terms)
        # Sample without replacement
        distractors = list(
            np.random.choice(possible_distractors, size=min(num_distractors, len(possible_distractors)), replace=False))
        # Combine and shuffle
        self.current_go_choices = list(true_terms.union(distractors))
        random.shuffle(self.current_go_choices)

        return {
            "sequence": self.current_sequence,
            "go_candidates": self.current_go_choices
        }

    def step(self, action):
        # amount of go terms
        selected_indices = np.nonzero(action)[0]
        selected_terms = [self.current_go_choices[i] for i in selected_indices]
        correct = set(selected_terms) & set(self.current_truth)
        amount_correct = len(correct)
        incorrect = set(selected_terms) - correct
        incorrectScore = len(incorrect) *.25
        total_true = len(set(self.current_truth))

        precision = amount_correct/len(selected_terms) if len(selected_terms) > 0 else 0
        recall = amount_correct/len(self.current_truth) if len(self.current_truth) > 0 else 0
        f1_score = 2 * (precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        reward = f1_score
        # Calculate percentage of correct GO terms guessed
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
        }, reward, done, {"percent_correct": percent_correct, "precision": precision,
                          "recall": recall, "selected_go": selected_terms}


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

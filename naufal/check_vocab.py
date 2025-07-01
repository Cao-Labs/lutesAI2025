import json

# === Load GO vocab
vocab_set = set(json.load(open("go_vocab.json")).keys())

# === Check test GO term coverage
covered = 0
total = 0

with open("matched_ids_with_go.txt") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        _, go_str = parts
        terms = [t for t in go_str.split(";") if t]
        total += len(terms)
        covered += sum(t in vocab_set for t in terms)

print(f"{covered}/{total} test GO terms are in the vocab")
print(f"Coverage: {100 * covered / total:.2f}%")

import torch

data = torch.load("AT10A_HUMAN.pt", map_location='cpu')
print(len(data))

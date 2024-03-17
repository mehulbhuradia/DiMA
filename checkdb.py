import pandas as pd
from torch.utils.data import DataLoader
from dataset import ProtienStructuresDataset


# Initialize your dataset
dataset = ProtienStructuresDataset()

print(len(dataset))

# Initialize a DataLoader
batch_size = 4  # Set your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test the DataLoader
for batch in dataloader:
    p, s = batch
    print(len(p[0]), len(p[1]))
    print(s[0].shape, s[1].shape)
    break
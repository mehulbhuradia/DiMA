import pandas as pd
from torch.utils.data import DataLoader
from dataset import ProtienStructuresDataset
from diffusion_utils.length_sampler import CustomLengthSampler


# Initialize your dataset
# dataset = ProtienStructuresDataset()

# print(len(dataset))

# # Initialize a DataLoader
# batch_size = 4  # Set your desired batch size
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test the DataLoader
# for batch in dataloader:
#     p, s = batch
#     print(len(p[0]), len(p[1]))
#     print(s[0].shape, s[1].shape)
#     break
import torch
dataset = ProtienStructuresDataset()
# Split the dataset into train and test
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

lensampler = CustomLengthSampler(valid_dataset, max_len=500)
print(lensampler.sample(10))
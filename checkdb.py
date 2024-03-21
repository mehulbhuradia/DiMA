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
dataset = ProtienStructuresDataset(use_cross_attention_on_context=True)
print(len(dataset))
dataset = ProtienStructuresDataset(use_cross_attention_on_context=False)
print(len(dataset))
# Split the dataset into train and test
# train_size = int(0.99 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# val_seqs = [t[0] for t in dataset]

# print(len(val_seqs))
# print(len(set(val_seqs)))

# print(val_seqs)

# print(valid_dataset[0])

# lensampler = CustomLengthSampler(valid_dataset, max_len=500)

# # lens, contexts = lensampler.sample(10)

# # print(type(contexts))
# # print(contexts.shape)

# print(lensampler.sample(1))
# print(lensampler.sample(1))
# print(lensampler.sample(1))
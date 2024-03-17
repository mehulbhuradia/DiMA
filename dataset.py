import os
import json
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd


class ProtienStructuresDataset(Dataset):
  def __init__(self, smiles_path='./ESP/smiles.pkl',csv_file='./ESP/esp_phylo_all.csv',max_len=500,min_len=50):
    with open(smiles_path, 'rb') as f:
        smiles_data = pickle.load(f)
    protien_df = pd.read_csv(csv_file)
    
    # Filter where 'Protein sequence' column length is between min_len and max_len
    protien_df = protien_df[(protien_df['Protein sequence'].str.len() >= min_len) &
            (protien_df['Protein sequence'].str.len() <= max_len)]
    
    dataset = {}
    count = 0
    for idx, row in protien_df.iterrows():
      if row['SMILES'] in smiles_data:
        dataset[row['Protein sequence']] = smiles_data[row['SMILES']]
    self.dataset = dataset
    
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    protien_seq = list(self.dataset.keys())[idx]
    smiles = list(self.dataset.values())[idx]
    return protien_seq, smiles
    
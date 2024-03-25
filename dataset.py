import os
import json
import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
from utils.preprocessing import load_fasta_file


class ProtienStructuresDataset(Dataset):
  def __init__(self, smiles_path='./ESP/smiles.pkl',csv_file='./ESP/esp_phylo_all.csv',max_len=500,min_len=50,use_cross_attention_on_context=True):
    with open(smiles_path, 'rb') as f:
        smiles_data = pickle.load(f)
    if ".csv" in csv_file:
      protein_df = pd.read_csv(csv_file)
    else:
      assert use_cross_attention_on_context == False, "If you are using a fasta file, you must set use_cross_attention_on_context to False"
      protein_list = load_fasta_file(csv_file)
      protein_df = pd.DataFrame({'Protein sequence': protein_list})
      # Adding dummy values of 0 for the other columns
      protein_df['Uniprot ID'] = 0
      protein_df['SMILES'] = 0

    # Filter where 'Protein sequence' column length is between min_len and max_len
    protein_df = protein_df[(protein_df['Protein sequence'].str.len() >= min_len) &
            (protein_df['Protein sequence'].str.len() <= max_len)]
    
    # Iterate over special letters and remove rows containing them in the 'Protein sequence' column    
    special_letters = "UZOXB"
    for letter in special_letters:
        protein_df = protein_df[~protein_df['Protein sequence'].str.contains(letter)]
    # If you want to reset the index after removing rows
    protein_df.reset_index(drop=True, inplace=True)
    
    dataset = [[],[]]
    for idx, row in protein_df.iterrows():
      if use_cross_attention_on_context:
        if row['SMILES'] in smiles_data:
          dataset[0].append(row['Protein sequence'])
          dataset[1].append(smiles_data[row['SMILES']])
      else:
        dataset[0].append(row['Protein sequence'])
    if not use_cross_attention_on_context:
      dataset[0]=list(set(dataset[0]))
      dataset[1] = [torch.zeros(1)] * len(dataset[0])
    self.dataset = dataset
    
  def __len__(self):
    return len(self.dataset[0])

  def __getitem__(self, idx):
    protien_seq = self.dataset[0][idx]
    smiles = self.dataset[1][idx]
    return protien_seq, smiles
    
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pickle
import pandas as pd
from tqdm import tqdm

SMILES_BERT = "DeepChem/ChemBERTa-77M-MTR"
smiles_reprs = {}
smiles_tokenizer = AutoTokenizer.from_pretrained(SMILES_BERT)
smiles_bert = AutoModelForMaskedLM.from_pretrained(SMILES_BERT)

def get_last_layer_repr(smiles):
    tokenizer = smiles_tokenizer
    model = smiles_bert
    key = "logits"

    tokens = tokenizer(
            smiles, 
            max_length=410, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt")
    tokens["input_ids"] = tokens["input_ids"]
    tokens["attention_mask"] = tokens["attention_mask"]
    last_layer_repr = model(**tokens)[key]
    return last_layer_repr.squeeze(0)



# Load dataset csv file


csv_file = "./ESP/esp_phylo_all.csv"

# Read CSV file into a DataFrame
df = pd.read_csv(csv_file)

smiles_set = set(df['SMILES'])

print(len(smiles_set))



# Initialize an empty dictionary to store the results
smiles_last_layer_repr_dict = {}

# Iterate over each SMILES string in the smiles_set and call the function
for smiles in tqdm(smiles_set, desc="Processing SMILES"):
    last_layer_repr = get_last_layer_repr(smiles)
    smiles_last_layer_repr_dict[smiles] = last_layer_repr

# Save the dictionary to a .pkl file
with open('./ESP/smiles.pkl', 'wb') as f:
    pickle.dump(smiles_last_layer_repr_dict, f)


# print(get_last_layer_repr("CCO").shape)

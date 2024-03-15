import pickle
import torch

# Load the pickled data
with open('./SMILES_ESP.pkl', 'rb') as f:
    data = pickle.load(f)

# Now `data` is a dictionary with PyTorch tensors as values
# You can access and work with these tensors as needed

for k, v in data.items():
    print(data[k].shape)  # This will print the shape of the tensor
    data[k] = v.squeeze(0).mean(dim=0)
    print(data[k].shape)  # This will print the shape of the tensor

# Save the modified dictionary back to a .pkl file
with open('./ESP/smiles_vec.pkl', 'wb') as f:
    pickle.dump(data, f)
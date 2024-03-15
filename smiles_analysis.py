import pickle
import torch

# Load the pickled data
with open('./ESP/smiles.pkl', 'rb') as f:
    data = pickle.load(f)

# Now `data` is a dictionary with PyTorch tensors as values
# You can access and work with these tensors as needed

keys = list(data.keys())

# For example, to access and print the first tensor:
first_tensor = data[keys[0]]  # Replace 'tensor_key' with the actual key used in the dictionary
print(first_tensor.shape)  # This will print the shape of the tensor
# torch.Size([1, 35, 600])

second_tensor = data[keys[1]]
print(second_tensor.shape)

# # You can also perform mathematical operations on tensors, like addition, multiplication, etc.
# # For example:
# result = torch.add(first_tensor, 1)  # Adds 1 to each element of the tensor
# print("Result after adding 1:", result)

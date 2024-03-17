import pickle

# Define the paths to the input files
file_paths = ['./ESP/smiles_1.pkl', './ESP/smiles_2.pkl', './ESP/smiles_3.pkl']

# Initialize an empty dictionary to store the merged data
big_dict = {}

# Load data from each pickle file and merge them into one big dictionary
for file_path in file_paths:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        big_dict.update(data)

# Define the path to save the merged dictionary
output_file_path = './ESP/smiles.pkl'

# Save the merged dictionary to a pickle file
with open(output_file_path, 'wb') as f:
    pickle.dump(big_dict, f)

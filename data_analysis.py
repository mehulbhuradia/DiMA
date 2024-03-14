import pandas as pd

def get_entries(csv_files):
    all_entries = []
    for csv_file in csv_files:
        # Read CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        # Append DataFrame to the list
        all_entries.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(all_entries, ignore_index=True)
    return combined_df

# wont work cuz i deleted the csv files

# Paths to your CSV files for test, validation, etc.
csv_files_1 = ['./ESP/train_val/ESP_train_df.csv', 
               './ESP/train_val/ESP_val_df.csv', 
               './ESP/train_val/ESP_test_df.csv']


# phylo contains normal dataset
csv_files_2 = ['./ESP/train_val_phylo/ESP_train_df.csv', 
               './ESP/train_val_phylo/ESP_val_df.csv', 
               './ESP/train_val_phylo/ESP_test_df.csv']

all_entries_df_1 = get_entries(csv_files_1)
all_entries_df_2 = get_entries(csv_files_2)

# Remove rows where output is 0
all_entries_df_1 = all_entries_df_1[all_entries_df_1['output'] != 0]
all_entries_df_2 = all_entries_df_2[all_entries_df_2['output'] != 0]


print("Column names:",all_entries_df_1.columns.tolist())

# Print min and max length of 'Protein sequence' before filtering
# print("Min length of 'Protein sequence' before filtering:")
# print("Dataset 1:", all_entries_df_1['Protein sequence'].str.len().min())
# print("Dataset 2:", all_entries_df_2['Protein sequence'].str.len().min())

# print("Max length of 'Protein sequence' before filtering:")
# print("Dataset 1:", all_entries_df_1['Protein sequence'].str.len().max())
# print("Dataset 2:", all_entries_df_2['Protein sequence'].str.len().max())

# Print min and max length of 'SMILES' before filtering
print("Min length of 'SMILES' before filtering:")
print("Dataset 1:", all_entries_df_1['SMILES'].str.len().min())
print("Dataset 2:", all_entries_df_2['SMILES'].str.len().min())

print("Max length of 'SMILES' before filtering:")
print("Dataset 1:", all_entries_df_1['SMILES'].str.len().max())
print("Dataset 2:", all_entries_df_2['SMILES'].str.len().max())


# Filter where 'Protein sequence' column length is between 128 and 254
filtered_df_1 = all_entries_df_1[(all_entries_df_1['Protein sequence'].str.len() >= 128) & 
                             (all_entries_df_1['Protein sequence'].str.len() <= 254)]

filtered_df_2 = all_entries_df_2[(all_entries_df_2['Protein sequence'].str.len() >= 128) & 
                             (all_entries_df_2['Protein sequence'].str.len() <= 254)]

print("Number of entries after filtering:")
print("Dataset 1:", len(filtered_df_1))
print("Dataset 2:", len(filtered_df_2))

# Drop 'molecule ID' and 'output' columns from filtered_df_1
filtered_df_1 = filtered_df_1.drop(columns=['molecule ID', 'output'])

# Drop 'molecule ID' and 'output' columns from filtered_df_2
filtered_df_2 = filtered_df_2.drop(columns=['molecule ID', 'output'])

# Save filtered DataFrames to CSV files
filtered_df_1.to_csv('esp.csv', index=False)
filtered_df_2.to_csv('esp_phylo.csv', index=False)

# Number of entries after filtering:
# Dataset 1: 1955
# Dataset 2: 27896
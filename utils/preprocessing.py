import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

def load_fasta_file(file_path):
    sequences = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences.append(str(record.seq))
    return sequences

def contains_special_letters(seq):
    special_letters = "UZOXB"
    for letter in special_letters:
        if letter in seq:
            return True
    return False

def trim_long_seqs(sequences, max_seq_length):
    processed_sequences = []
    for seq in sequences:
        # TODO: Implement preprocessing
        # Convert to upper case, check for invalid characters

        # Trim sequences longer than max_seq_length
        if len(seq) > max_seq_length:
            start = random.randint(0, len(seq) - max_seq_length)
            seq = seq[start:start + max_seq_length]

        if contains_special_letters(seq):
            continue

        processed_sequences.append(seq)
    
    return processed_sequences

def remove_short_seqs(sequences, min_seq_length):
    processed_sequences = []
    for seq in sequences:
        if len(seq) < min_seq_length:
            continue

        processed_sequences.append(seq)
    
    return processed_sequences

def remove_long_seqs(sequences, max_seq_length):
    processed_sequences = []
    for seq in sequences:
        if len(seq) > max_seq_length:
            continue

        processed_sequences.append(seq)
    
    return processed_sequences

def preprocess_sequences(file_path, min_seq_length=128, max_seq_length=254):
    sequences = load_fasta_file(file_path)
    sequences = remove_short_seqs(sequences, min_seq_length)
    sequences = remove_long_seqs(sequences, max_seq_length)
    for seq in sequences:
        if contains_special_letters(seq):
            sequences.remove(seq)
    return sequences

def preprocess_fasta(file_path, min_seq_length=50, max_seq_length=500):
    sequences = []
    with open(file_path, "r") as fasta_file:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        for record in tqdm(records, desc="Processing sequences", unit="sequence"):
            s = str(record.seq)
            if len(s) < min_seq_length or len(s) > max_seq_length or contains_special_letters(s):
                continue
            sequences.append(SeqRecord(Seq(s)))
    return sequences

def make_train_val(filepath, train_size=0.9):
    sequences = preprocess_fasta(filepath)
    random.shuffle(sequences)
    train_size = int(len(sequences) * train_size)
    train = sequences[:train_size]
    val = sequences[train_size:]
    return train, val

def preprocess_fasta_trim(file_path, min_seq_length=128, max_seq_length=254):
    sequences = []
    with open(file_path, "r") as fasta_file:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        for record in tqdm(records, desc="Processing sequences", unit="sequence"):
            s = str(record.seq)
            if len(s) < min_seq_length or contains_special_letters(s):
                continue
            if len(s) > max_seq_length:
                start = random.randint(0, len(s) - max_seq_length)
                s = s[start:start + max_seq_length]
            sequences.append(SeqRecord(Seq(s)))
    return sequences

def make_train_val_trim(filepath, train_size=0.9):
    sequences = preprocess_fasta_trim(filepath)
    random.shuffle(sequences)
    train_size = int(len(sequences) * train_size)
    train = sequences[:train_size]
    val = sequences[train_size:]
    return train, val

def save_fasta(sequences, output_file):
    SeqIO.write(sequences, output_file, "fasta")
    print(f"FASTA file '{output_file}' created successfully.")


if __name__ == "__main__":
    # Example usage
    # train, val = make_train_val("data/AFDBv4_90.fasta")
    # save_fasta(train, "data/AFDB/AFDBv4_90.128-254-train.fasta")
    # save_fasta(val, "data/AFDB/AFDBv4_90.128-254-valid.fasta")
    # print(f"Train size: {len(train)}")
    # print(f"Validation size: {len(val)}")
    train, val = make_train_val("data/uniprot_sprot.fasta")
    save_fasta(train, "data/uniprot_500/uniprot_500-train.fasta")
    save_fasta(val, "data/uniprot_500/uniprot_500-valid.fasta")
    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(val)}")
    #     Train size: 400727
    # Validation size: 44526
    # train, val = make_train_val_trim("data/uniprot_sprot.fasta")
    # save_fasta(train, "data/uniprot_trim/uniprot_trim-train.fasta")
    # save_fasta(val, "data/uniprot_trim/uniprot_trim-valid.fasta")
    # print(f"Train size: {len(train)}")
    # print(f"Validation size: {len(val)}")
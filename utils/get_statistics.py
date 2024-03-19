import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import create_config
from dataset import ProtienStructuresDataset
from encoders import ESM2EncoderModel
from utils import load_fasta_file


def get_loader(config,  batch_size):
    train_dataset = ProtienStructuresDataset(smiles_path=config.data.smiles_path, csv_file=config.data.csv_file, max_len=config.data.max_sequence_len, min_len=config.data.min_sequence_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    return train_loader


def compute_mean_std(
        config,
        encoder,
        model_name, 
        dataset_name,
):
    sum_ = None
    sqr_sum_ = None
    num = 0
    batch_size = 512

    train_loader = get_loader(
        config=config,
        batch_size=batch_size
    )
    T = tqdm(train_loader)

    for i, input in enumerate(T):
        with torch.no_grad():
            output, _ = encoder.batch_encode(input[0])

        cur_sum = torch.sum(output, dim=[0, 1])
        cur_sqr_sum = torch.sum(output ** 2, dim=[0, 1])
        cur_num = output.shape[0] * output.shape[1]

        sum_ = cur_sum if sum_ is None else cur_sum + sum_
        sqr_sum_ = cur_sqr_sum if sqr_sum_ is None else cur_sqr_sum + sqr_sum_
        num += cur_num

        mean = sum_[:3] / num
        std = torch.sqrt(sqr_sum_[:3] / num - mean ** 2)
        T.set_description(f"mean: {[m.item() for m in mean]}, std2: {[s.item() for s in std]}")

    mean = sum_ / num
    std = torch.sqrt(sqr_sum_ / num - mean ** 2)

    folder_path = f"./data/{dataset_name}/"
    os.makedirs(folder_path, exist_ok=True)
    torch.save(mean, f'{folder_path}/encodings-{model_name}-mean.pt')
    torch.save(std, f'{folder_path}/encodings-{model_name}-std.pt')


if __name__ == "__main__":
    config = create_config()
    encoder = ESM2EncoderModel(
        config.model.hg_name, 
        device="cuda:0", 
        decoder_path=None, 
        max_seq_len=config.data.max_sequence_len,
        enc_normalizer=None,
    )

    compute_mean_std(
        config,
        encoder,
        model_name=config.model.hg_name_hash,
        dataset_name=config.data.dataset
    )

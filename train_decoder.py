import os
import torch
from dataset import ProtienStructuresDataset
import wandb
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from config import create_config
from encoders import ESM2EncoderModel
from utils import load_fasta_file


def reconstruction_loss(target, prediction_scores, mask):
    if mask is None:
        return cross_entropy(
            input=prediction_scores.view(-1, prediction_scores.shape[-1]),
            target=target.view(-1),
        )

    ce_losses = cross_entropy(
        input=prediction_scores.view(-1, prediction_scores.shape[-1]),
        target=target.view(-1),
        reduce=False,
    )
    ce_losses = ce_losses * mask.reshape(-1)
    ce_loss = torch.sum(ce_losses) / torch.sum(mask)
    return ce_loss


def get_loaders(config, batch_size):
    
    dataset = ProtienStructuresDataset(smiles_path=config.data.smiles_path, csv_file=config.data.csv_file, max_len=config.data.max_sequence_len, min_len=config.data.min_sequence_len)
    
    # Split the dataset into train and test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size
    )

    return train_loader, valid_loader


def loss_step(input, encoder, decoder, eval=False):
    X = input[0]
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        latent, tokenized_X = encoder.batch_encode(X)

    if not eval:
        sigma = 0.2
        eps = torch.randn_like(latent) * sigma
        latent = latent + eps
    
    targets = tokenized_X["input_ids"]
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = decoder(latent)
    loss = reconstruction_loss(targets, logits, mask=None)
    
    tokens = logits.argmax(dim=-1)
    acc = torch.mean((targets == tokens) * 1.)

    return loss, acc


def train(config, encoder, decoder, exp_name):
    total_number_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Num params: {total_number_params}")
    
    batch_size = 512

    train_loader, valid_loader = get_loaders(
        config=config,
        batch_size=batch_size
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=5e-5,
        weight_decay=0.001,
        betas=(0.9, 0.98),
    )

    eval_freq = 1000
    step = 0
    epochs = 1000
    
    checkpoints_folder = './checkpoints/'
    os.makedirs(checkpoints_folder, exist_ok=True)

    best_val_loss = float('inf')
    early_stopping_patience = 50

    for _ in range(epochs):
        decoder.train()

        for X in tqdm(train_loader):
            loss, acc = loss_step(
                input=X,
                encoder=encoder,
                decoder=decoder
            )
       
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=1.0
            )
            optimizer.step()

            wandb.log({f'train loss': loss.item()}, step=step)
            wandb.log({f'train accuracy': acc.item()}, step=step)

            step += 1

            if step % eval_freq == 0:
                decoder.eval()
                for X in tqdm(valid_loader):
                    with torch.no_grad():
                        loss, acc = loss_step(
                            input=X,
                            encoder=encoder,
                            decoder=decoder,
                            eval=True
                        )
                decoder.train()

                wandb.log({f'valid loss': loss.item()}, step=step)
                wandb.log({f'valid accuracy': acc.item()}, step=step)
                if loss < best_val_loss:
                    early_stopping_counter = 0
                    best_val_loss = loss
                    name = os.path.join(checkpoints_folder, f"decoder-{config.model.hg_name_hash}-{config.data.dataset}--{step}.pth")
                    decoder.eval()
                    torch.save(
                        {
                            "decoder": decoder.state_dict(),
                        },
                        name
                    )
                    print(f"Save model to: {name}")
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"Early stopping at step {step}")
                        return

    


if __name__ == "__main__":
    config = create_config()
    encoder = ESM2EncoderModel(
        config.model.hg_name, 
        device="cuda:0", 
        decoder_path=None, 
        max_seq_len=config.data.max_sequence_len,
        enc_normalizer=None,
    )

    decoder = encoder.decoder.train()

    exp_name = f"decoder-{config.model.hg_name_hash}-{config.data.dataset}"
    wandb.init(project=config.project_name, name=exp_name, mode="online")
    train(config, encoder, decoder, exp_name=exp_name)

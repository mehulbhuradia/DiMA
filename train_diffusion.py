import os
import torch
print(torch.__version__)
from diffusion_utils.diffusion_holder import DiffusionRunner
from utils.util import set_seed
from config import create_config

print("PyTorch version:", torch.__version__)

if __name__ == '__main__':
    config = create_config()
    config.checkpoints_prefix = "DiMA-AFDB"
    config.training.batch_size_per_gpu = config.training.batch_size 
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.project_name = 'proteins'

    seed = config.seed
    set_seed(seed)

    print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed
    set_seed(seed)
    
    diffusion.train(
        project_name=config.project_name,
        experiment_name=config.checkpoints_prefix
    )

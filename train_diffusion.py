import os
import torch
from diffusion_utils.diffusion_holder import DiffusionRunner
from utils.util import set_seed
from config import create_config
import time

if __name__ == '__main__':
    config = create_config()
    timstamp = str(time.time()).replace(".","")
    config.training.checkpoints_folder = './checkpoints/'+ timstamp + '/'
    config.checkpoints_prefix = "DiMA_CROSS"+timstamp+"_"+config.data.dataset+"_"+config.model.hidden_size
    config.training.batch_size_per_gpu = config.training.batch_size 
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.project_name = 'proteins'

    seed = config.seed
    set_seed(seed)

    print(config)

    diffusion = DiffusionRunner(config, latent_mode=config.model.embeddings_type)

    seed = config.seed
    set_seed(seed)
    
    diffusion.train()

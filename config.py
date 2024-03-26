import ml_collections
from transformers import BertConfig

# 320 for 8M, 640 for 150M
model_size = 320
use_cross_attention_on_context = True
training_iters = 1_000_000

if use_cross_attention_on_context:
    max_sequence_len = 500 # 256 or 500
else:
    max_sequence_len = 256


if max_sequence_len == 500:
    min_sequence_len = 50
    dataset_path = './ESP/esp_phylo_all.csv'
else:
    min_sequence_len = 128
    dataset_path = './data/AFDBv4_90.fasta'

if use_cross_attention_on_context:
    learning_rate = 5e-5
    checkpoint_freq = 10_000
    eval_freq = 25_000
else:
    learning_rate = 2e-4
    checkpoint_freq = 100_000
    eval_freq = 150_000

def create_config():
    config = ml_collections.ConfigDict()
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.
    optim.linear_warmup = 1_000
    optim.lr = learning_rate
    optim.min_lr = 5e-6
    optim.warmup_lr = 0.
    optim.weight_decay = 0.01
    optim.beta_1 = 0.9
    optim.beta_2 = 0.98
    optim.eps = 1e-6

    training = config.training = ml_collections.ConfigDict()
    training.training_iters = training_iters
    training.checkpoint_freq = checkpoint_freq
    training.eval_freq = eval_freq
    training.batch_size = 32

    training.ode_sampling = False
    training.checkpoints_folder = './checkpoints/'
    config.checkpoints_prefix = ''

    loss = config.loss = ml_collections.ConfigDict()
    loss.ce_coef = 0.

    refresh = config.refresh = ml_collections.ConfigDict()
    refresh.true = False
    refresh.prefix = ""

    validation = config.validation = ml_collections.ConfigDict()
    validation.batch_size = training.batch_size
    validation.validation_iters = int(10_000 / validation.batch_size)
    validation.num_gen_texts = 2048

    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 100
    sde.coef_d = 10.
    sde.ode_sampling = False
    sde.scheduler = "sd"

    model = config.model = ml_collections.ConfigDict()
    model.ema_rate = 0.9999
    model.embeddings_type = "encodings"
    model.dif_enc_type = "base"
    model.prediction = "x_0"
    model.loss = "L_x_0"
    # model.hidden_size = 640
    # model.hg_name = "facebook/esm2_t30_150M_UR50D"
    # model.hg_name_hash = "esm2-150M"
    model.hidden_size = model_size
    if model.hidden_size == 640:
        model.hg_name = "facebook/esm2_t30_150M_UR50D"
        model.hg_name_hash = "esm2-150M"
    else:
        model.hg_name = "facebook/esm2_t6_8M_UR50D"
        model.hg_name_hash = "esm2-8M"

    data = config.data = ml_collections.ConfigDict()
    data.max_sequence_len = max_sequence_len
    data.min_sequence_len = min_sequence_len
    
    # Dataset Path
    data.csv_file = dataset_path
    
    data.smiles_path = './ESP/smiles.pkl'
    data.dataset = "uniprot_500"

    if max_sequence_len == 500:
        data.dataset = "uniprot_500"
    elif max_sequence_len == 256:
        data.dataset = "AFDB"
    
    if data.dataset == "uniprot_500":
        data.decoder_epoch = "4000"
    elif data.dataset == "uniprot":
        data.decoder_epoch = "203000"
    elif data.dataset == "uniprot_trim":
        data.decoder_epoch = "53000"
    elif data.dataset == "AFDB":
        data.decoder_epoch = "340000"
    
    
    data.enc_mean = f"./data/{data.dataset}/encodings-{model.hg_name_hash}-mean.pt"
    data.enc_std = f"./data/{data.dataset}/encodings-{model.hg_name_hash}-mean.pt"
    
    config.decoder_path = f"./checkpoints/decoder-{config.model.hg_name_hash}-{config.data.dataset}--{config.data.decoder_epoch}.pth"
    config.seed = 0
    config.ddp = False
    config.use_self_cond = True
    config.bert_config = bert_config
    config.project_name = "proteins_dif"

    return config


bert_config = BertConfig(**{
    "hidden_size": model_size,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 16,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "attention_probs_dropout_prob": 0.1,
    "layer_norm_eps": 1e-12,
    "model_type": "bert",
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.6.0.dev0",
    "is_decoder": False,
    "cross_context_dim": 600,
    "cross_gated_ff": True,
    "use_cross_attention_on_context": use_cross_attention_on_context,
})
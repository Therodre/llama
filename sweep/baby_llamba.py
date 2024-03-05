# Ugh, Ugh, Ugh

from datetime import datetime


out_dir = "out/BL"
eval_interval = 10
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
vocab_source = (
    "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
)
vocab_size = 6070  # the simple tokenizer has 6070 chars |32000 the Llama 2 tokenizer has 32K tokens
# model
dim = 256
n_layers = 6
n_heads = 4
n_kv_heads = 4
multiple_of = 32
dropout = 0.0
loss_normalization = True
hybrid = True
# adamw optimizer
gradient_accumulation_steps = 16  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = "float16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# ssm layers config
dt_rank = "auto"
d_state = 16  # N in paper/comments
expand_factor = 2  # E in paper/comments
d_conv = 4
dt_min = 0.001
dt_max = 0.1
dt_init = "random"  # "random" or "constant"
dt_scale = 1.0
# dt_init_floor=1e-4,
bias = False
conv_bias = True
pscan = True  # use parallel scan mode or sequential mode when training

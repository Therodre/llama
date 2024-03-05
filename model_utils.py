import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs, MambaConfig


def build_model(out_dir, device, strict=True):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model_args = dict()

    for k in [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "vocab_size",
        "multiple_of",
        "max_seq_len",
        "dropout",
        "ssm_config",
        "loss_normalization",
        "hybrid",
    ]:

        model_args[k] = checkpoint_model_args[k]
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict)
    model.to(device)
    return model

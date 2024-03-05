import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from model import Transformer, ModelArgs, MambaConfig


class Evaluator:
    def __init__(self, model, iter_batches) -> None:
        self.model = model
        self.iter_batches = iter_batches
        self.ctx = nullcontext()  # no ddp no fsdp

    @torch.no_grad()
    def evaluate(self, steps=1):
        model.eval()
        batch_iter = self.iter_batches
        loss_record = [torch.zeros(size=())]
        for k in range(steps):
            X, Y = next(batch_iter)
            with self.ctx:
                logits = model(X, Y)
                loss = model.last_loss
            loss_record.append(loss.item())
            if k % 100 == 0 and k > 10:
                print(f"== MEAN LOSS : {sum(loss_record)/len(loss_record)}\n")

        print(
            f"------------FINAL LOSS --------------\n {sum(loss_record)/len(loss_record)}"
        )


if __name__ == "__main__":
    from model_utils import build_model
    from tinystories import Task
    from argparse import ArgumentParser
    from sweep.baby_llamba import *

    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Input the directory of the model (not including the model.pth in the path)",
    )
    args = parser.parse_args()
    out_dir = args.model_dir
    device = "cuda"
    batch_size = 16
    wandb_log = False
    vocab_source = "enwik"
    with_dist = "False"
    dist_coeff = 0.0
    # build model
    model = build_model(out_dir, device)
    model.register_buffer("dist_coef", torch.tensor(0.0))
    iter_batches = partial(
        Task.iter_batches,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device=device,
        num_workers=0,
    )
    eval_iter_batches = iter_batches(split="test")
    evaluator = Evaluator(model, eval_iter_batches)
    evaluator.evaluate(100000)

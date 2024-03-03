import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
import numpy as np
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from tinystories import PretokDataset, Task
from export import model_export
from model_utils import build_model
from model import Transformer, ModelArgs, MambaConfig


TEACHER_CACHE_DIR = "/storage/enwik8/teacher/"
TEACHER_MODEL_DIR = "out/teacher"


class DistillationCacheWriter:
    def __init__(self, model, cache_dir, layers_to_teach, ctx) -> None:
        self.teacher_model: Transformer = model
        self.cache_dir = cache_dir
        self.layers_to_teach = layers_to_teach
        self.ctx = ctx

    @torch.no_grad()
    def create_cache(self, iter_batches, steps, data_source) -> None:
        self.teacher_model.eval()
        all_pred = []
        for k in range(steps):
            X, _ = next(iter_batches)
            with self.ctx:
                predictions = self.teacher_model.teaching_forward(
                    X, self.layers_to_teach
                )
            predictions = predictions.detach()
            predictions.requires_grad = False
            predictions = predictions.to("cpu")
            all_pred.append(predictions)
        all_pred = np.array(all_pred)
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = os.path.join(
            self.cache_dir,
            f"teacher_numpies_{now}.bin",
        )
        with open(filename, "wb") as f:
            f.write(all_pred.tobytes())
        metadata_file_name = os.path.join(
            self.cache_dir,
            f"metadata.json",
        )
        with open(metadata_file_name, "w") as f:
            d = {
                "date": datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                "tensor_shape": all_pred[0].shape,
                "number of samples": len(all_pred),
                "data": data_source,
            }
            f.write(json.dumps(d))


class DistillDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir, max_seq_len):
        super().__init__()


if __name__ == "__main__":
    from sweep.enwik_baby_llamba import *

    model = build_model(TEACHER_MODEL_DIR, device="cuda")
    batch_size = 16
    iter_batches = Task.iter_batches(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        vocab_source=vocab_source,
        device=device,
        num_workers=0,
        split="train",
    )

    cache_writer = DistillationCacheWriter(
        model=model,
        cache_dir=TEACHER_CACHE_DIR,
        layers_to_teach=[0, 2, 4],
        ctx=nullcontext(),
    )
    cache_writer.create_cache(
        iter_batches=iter_batches, data_source=(vocab_source, "train"), steps=1000
    )

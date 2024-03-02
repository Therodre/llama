# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import pickle
import struct
import argparse
import pathlib
from typing import List

from sentencepiece import SentencePieceProcessor
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


TOKENIZER_MODEL = "tokenizer.model"  # the llama sentencepiece tokenizer model
CHARACTER_FILE = "/home/rod/storage/enwik8/char.pkl"


class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        # print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):

        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = "\n<s>\n"
            elif i == self.eos_id:
                t = "\n</s>\n"
            t = t.replace("▁", " ")  # sentencepiece uses this character as whitespace
            b = t.encode("utf-8")  # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace(".model", ".bin")
        with open(tokenizer_bin, "wb") as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)


class SimpleTokenizer(PreTrainedTokenizer):
    # mostly coming from here https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py
    def __init__(self, **kwargs):
        with open(CHARACTER_FILE, "rb") as f:
            characters = pickle.load(f)
        self.characters = characters
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[PAD]": 3,
            "[EOS]": 4,
            "[UNK]": 5,
            **{ch: i + 6 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_prefix_space=False,
            **kwargs,
        )

    def get_vocab(self):
        return self._vocab_str_to_int

    @property
    def vocab_size(self):
        return len(self._vocab_str_to_int)

    @property
    def bos_id(self):
        # FIXME: Dude... please
        return 2

    @property
    def eos_id(self):
        # FIXME: Dude... please
        return 4

    @property
    def pad_id(self):
        # FIXME: Dude... please
        return 3

    @property
    def unk_id(self):
        # FIXME: Dude... please
        return 5

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = [self._vocab_str_to_int.get(u, self.unk_id) for u in s]
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return "".join([self._vocab_int_to_str.get(u) for u in t])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    tok = SimpleTokenizer()
    s = "Tu es vraiment un gros legume, ';[]9876543´éeee\]=0-0890"
    print(s)
    t = tok.encode(s, True, True)
    print(t)
    print(tok.decode(t))
    print(tok.vocab_size)
    # parser.add_argument(
    #     "-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer "
    # )
    # args = parser.parse_args()

    # t = Tokenizer(args.tokenizer_model)
    # t.export()

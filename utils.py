import requests
import os
from typing import Dict
import torch
import numpy as np


class Trainer:
    def __init__(
        self,
        # training parameters
        max_iters: int = 600000,
        # batching parameters
        batch_size: int = 12,
        block_size: int = 1024,
        # evaluation parameters
        eval_interval: int = 2000,
        eval_iters: int = 200,
    ):
        self.max_iters = max_iters

        # load data
        train_ids, val_ids = self.prepare_data()
        self.train_ids = torch.from_numpy(train_ids)
        self.val_ids = torch.from_numpy(val_ids)

        # set batching parameters
        self.batch_size = batch_size
        self.block_size = block_size

        # evaluation parameters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

    def train_step(self, model, optimizer: torch.optim.Optimizer):
        model.train()
        input, target = self.get_batch('train')
        loss = model.compute_loss(model(input), target)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    def get_batch(self, split: str) -> torch.Tensor:
        '''Returns a (B, T) shaped tensor'''
        data = self.train_ids if split == 'train' else self.val_ids
        n = len(data)
        start_id = torch.randint(0, n - self.block_size, size=(self.batch_size,))
        # Say (55, 56, 57, 58) is the block size
        xb = torch.stack([data[s:s+self.block_size] for s in start_id])
        yb = torch.stack([data[s+1:s+self.block_size+1] for s in start_id])
        return xb, yb

    @torch.no_grad()
    def estimate_loss(self, model) -> Dict[str, float]:
        out = {}
        model.eval()
        for split in ['train', 'val']:
            loss = 0
            for _ in range(self.eval_iters):
                input, target = self.get_batch(split)
                loss += model.compute_loss(model(input), target).item()
            out[split] = loss / self.eval_iters # report average loss
        model.train()
        return out

    def train(self, model, optimizer):
        for iter in range(self.max_iters):
            # run training
            self.train_step(model, optimizer)

            if iter % self.eval_interval == 0:
                # run evaluation
                output = self.estimate_loss(model)
                print(f"Iter: {iter}, Train Loss: {output['train']:.4f}, Val Loss: {output['val']:.4f}")

    def generate(self, model, max_len: int = 1000):
        input = torch.zeros((1, 1), dtype=torch.int64) # (1, 1)
        output = model.generate(input, max_len=max_len)[0].tolist()
        return self.decode(output)

    def prepare_data(self):
        input_file_path = 'data/input.txt'
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, 'r') as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")

        # get all the unique characters that occur in this text
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        print(f"vocab size: {self.vocab_size:,}")

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        # create the train and test splits
        n = len(data)
        train_data = data[:int(n*0.9)]
        val_data = data[int(n*0.9):]

        train_ids = np.array(self.encode(train_data))
        val_ids = np.array(self.encode(val_data))

        return train_ids, val_ids

    def decode(self, tokens):
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[i] for i in tokens])

    def encode(self, tokens):
        # encoder: take a string, output a list of integers
        return [self.stoi[c] for c in tokens]

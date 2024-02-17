import torch
import torch.nn.functional as F
from torch import nn
from utils import Trainer

# Some global parameters
# ----------------------
block_size = 32
dropout = 0.2
n_layer = 6
max_iters = 10000
batch_size = 12
eval_interval = 2000
eval_iters = 200
eval_interval = 2000
# ----------------------

class SingleHeadAttention(nn.Module):
    '''Single Head Attention Implementation'''
    def __init__(self, n_embd: int, head_dim: int):
        super().__init__()
        self.key_layer = nn.Linear(n_embd, head_dim, bias=False)
        self.query_layer = nn.Linear(n_embd, head_dim, bias=False)
        self.value_layer = nn.Linear(n_embd, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # shape of x: (B, T, n_embd)
        _, T, C = x.shape
        q = self.query_layer(x) # (B, T, head_dim)
        k = self.key_layer(x) # (B, T, head_dim)
        v = self.value_layer(x) # (B, T, head_dim)

        # compute attention matrix
        wei = torch.bmm(q, k.transpose(-2, -1)) * C ** (-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # apply attention to value
        # (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim)
        return torch.bmm(wei, v)
    

class MultiHeadAttention(nn.Module):
    '''Multi-Head Attention Implementation'''
    def __init__(self, n_embd: int, head_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.key_layer = nn.Linear(n_embd, head_dim, bias=False)
        self.query_layer = nn.Linear(n_embd, head_dim, bias=False)
        self.value_layer = nn.Linear(n_embd, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.proj = nn.Linear(head_dim, head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # shape of x: (B, T, n_embd)
        B, T, _ = x.shape
        H = self.n_heads
        C = self.head_dim
        # (B, H, T, head_dim // H)
        q = self.query_layer(x).view(B, T, H, -1).transpose(1, 2)
        k = self.key_layer(x).view(B, T, H, -1).transpose(1, 2)
        v = self.value_layer(x).view(B, T, H, -1).transpose(1, 2)
        assert q.shape == (B, H, T, C // H)

        # compute attention matrix
        wei = torch.einsum('ijkl,ijlm->ijkm', q, k.transpose(-2, -1)) * C ** (-0.5) # (B, H, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # apply attention to value
        # (B, H, T, T) @ (B, H, T, head_dim) -> (B, H, T, head_dim)
        out = torch.einsum('ijkl,ijlm->ijkm', wei, v).transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.proj(out))
    

class BiGramModel(nn.Module):
    '''Bi-Gram Model
    Prediction based on bi-gram model.
    '''
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input: torch.Tensor):
        # input: (B, T)
        logits = self.embed(input) # (B, T, C)
        return logits

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        C = logits.shape[2]
        return F.cross_entropy(
            logits.view(-1, C), # pytorch needs channels as the last dimension
            target.view(-1),
        )

    def generate(self, input: torch.Tensor, max_len: int = 1000):
        '''
        Input Shape: (B, T)
        Adds additional tokens to the input such that
        output has shape (B, T + max_len)
        '''
        output = input # (B, T)
        for _ in range(max_len):
            logits = self(output) # (B, T + i, C)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (B, C)
            next_token = torch.multinomial(probs, 1) # (B, 1)
            output = torch.cat([output, next_token], dim=1) # (B, T + i + 1)

        return output


class BiGramWithAttention(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, head_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attn = MultiHeadAttention(n_embd, head_dim, block_size)
        self.ln = nn.LayerNorm(head_dim)
        self.lm_head = nn.Linear(head_dim, vocab_size)

    def forward(self, input: torch.Tensor):
        # input: (B, T)
        B, T = input.shape
        token_emb = self.token_embedding(input) # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=input.device)) # (T, n_embd)
        x = token_emb + pos_emb # (B, T, n_embd)
        x = self.attn(x) # (B, T, head_dim)
        x = self.ln(x) # (B, T, head_dim)
        return self.lm_head(x) # (B, T, vocab_size)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        C = logits.shape[2]
        return F.cross_entropy(
            logits.view(-1, C), # pytorch needs channels as the last dimension
            target.view(-1),
        )

    def generate(self, input: torch.Tensor, max_len: int = 1000):
        '''
        Input Shape: (B, T)
        Adds additional tokens to the input such that
        output has shape (B, T + max_len)
        '''
        output = input # (B, T)
        for _ in range(max_len):
            logits = self(output[:, -block_size:]) # (B, T, C)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (B, C)
            next_token = torch.multinomial(probs, 1) # (B, 1)
            output = torch.cat([output, next_token], dim=1) # (B, T + 1)

        return output


class FeedForward(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(head_dim, 4 * head_dim),
            nn.ReLU(),
            nn.Linear(4 * head_dim, head_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, head_dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, head_dim, n_heads)
        self.ff = FeedForward(head_dim)
        self.ln1 = nn.LayerNorm(head_dim)
        self.ln2 = nn.LayerNorm(head_dim)

    def forward(self, x):
        # Note: Residual Connection
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class BiGramWithBlocks(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, head_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attn = nn.Sequential(*[Block(n_embd, head_dim, n_heads=4)] * n_layer)
        self.ln = nn.LayerNorm(head_dim)
        self.lm_head = nn.Linear(head_dim, vocab_size)

    def forward(self, input: torch.Tensor):
        # input: (B, T)
        B, T = input.shape
        token_emb = self.token_embedding(input) # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=input.device)) # (T, n_embd)
        x = token_emb + pos_emb # (B, T, n_embd)
        x = self.attn(x) # (B, T, head_dim)
        x = self.ln(x) # (B, T, head_dim)
        return self.lm_head(x) # (B, T, vocab_size)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        C = logits.shape[2]
        return F.cross_entropy(
            logits.view(-1, C), # pytorch needs channels as the last dimension
            target.view(-1),
        )

    def generate(self, input: torch.Tensor, max_len: int = 1000):
        '''
        Input Shape: (B, T)
        Adds additional tokens to the input such that
        output has shape (B, T + max_len)
        '''
        output = input # (B, T)
        for _ in range(max_len):
            logits = self(output[:, -block_size:]) # (B, T, C)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (B, C)
            next_token = torch.multinomial(probs, 1) # (B, 1)
            output = torch.cat([output, next_token], dim=1) # (B, T + 1)

        return output


if __name__ == "__main__":
    trainer = Trainer(
        max_iters=10000,
        batch_size=12,
        block_size=block_size,
        eval_interval=2000,
        eval_iters=200,
    )
    # model = BiGramModel(vocab_size=vocab_size)
    model = BiGramWithBlocks(trainer.vocab_size, n_embd=32, head_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # train the model
    trainer.train(model, optimizer)
    # generate some text from empty sequence
    print (trainer.generate(model, max_len = 1000))

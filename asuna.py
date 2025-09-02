import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import datetime
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu' # use cuda(gpu) if not available use cpu
print (device)

block_size = 128 # how many blocks can the model see at once
n_embd = 384 # creates a 384 attribute embedding_vector
n_layer = 8
n_head = 8
dropout = 0.25 # 20% neurons will be turned off to prevent overfitting 

chat_logs = "log_files/chat_logs.txt"

chars = ""
with open ('vocab.txt', 'r', encoding='utf-8') as f: # use text and encode it with utf-8
        text = f.read() # let the script read the text
        chars = sorted(list(set(text))) # sort all used characters in text
    
vocab_size = len(chars) # sets the vocab_size to the number of characters in chars

string_to_int = { ch:i for i,ch in enumerate(chars) } # make string of characters into intigers (full numbers)
int_to_string = { i:ch for i,ch in enumerate(chars) } # make intigers into string of characters
encode = lambda s: [string_to_int[c] for c in s] # encode the characters into the intigers
decode = lambda l: ''.join([int_to_string[i] for i in l]) # decode the intigers into characters

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # converting n_embd into head size 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # registers the no looking ahead rule and the buffer saves computation because it doesnt need to be initialized every time ig

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of isze (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        wei = wei.masked_fill(~mask, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values (aggregate: add multiple elements into a single entity)
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # makes 4 heads running in parallel
        self.proj = nn.Linear(head_size * num_heads, n_embd) # projects the head_size times the num_heads into n_embd
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatinate the head features along the last dimension of (B, T, C) -> (B, T, [h1, h1, h1, h1, h2, h2, h2, h2...])
        out = self.dropout(self.proj(out)) # project concatenated heads back down to embedding size + apply dropout
        return out
        
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # expands embedding size by 4
            nn.ReLU(),                     # adds a non-linearity so the model to learn more complex transformations
            nn.Linear(4 * n_embd, n_embd), # shrink the embedding size back down
            nn.Dropout(dropout)            # apply dropout for regularization 
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module): # decoder 
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number if heads want
        super().__init__()
        head_size = n_embd // n_head # number of features that each head captures (96 each)
        self.sa = MultiHeadAttention(n_head, head_size) # sa: self attention
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x) # self attention first
        x= self.ln1(x + y) # layer norm (residual connection add the norm) second 
        y = self.ffwd(x) # feed forward third 
        x = self.ln2(x + y) # last layer norm (residual connection add the norm) again 
        return x
        
class AsunaLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # This creates a table that turns each word into a prediction for the next word.
        # It's like saying: "If I see word X, what word should come next?"
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #the same as token embedding but not every char has its own embed but each letter index has its own embed
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # how many decoder blocks we have running sequentialy / layers 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        # transforming the decoders output into something that softmax can work with 
        self.lm_head = nn.Linear(n_embd, vocab_size) #language modeling head

        self.apply(self._init_weights)

    # this shit just helps with stable learning of the model i wont go deeper cause it is fkin confusing 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, index, targets=None):
        B, T = index.shape
        #idx and targets are bith (B,T) tensor of intigers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        # how long is T, it creates T number in indecies and give each a different n_embd vector (a little lookup table)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,C,vocab_size)
        
        if targets is None:
            # If we're just using the model (not training), no need for a loss.
            loss = None
        else:
            # We're training the model, so calculate how wrong the guesses were.
            # First, flatten the logits and targets to make them easier to compare.
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)         # make logits 2D: (all tokens, vocab_size)
            targets = targets.view(B*T)          # flatten targets too: (all tokens)
            # Now calculate how far off the model’s guesses were.
            loss = F.cross_entropy(logits, targets)

        # Return both the guesses and the loss (loss is None during generation)
        return logits, loss

    def generate(self, index, max_new_tokens, temperature=1.0, top_p=None):
        self.eval()
        with torch.no_grad():
            # interpret position embedding size as block_size (max context length)
            block_size = self.position_embedding_table.num_embeddings

            for _ in range(max_new_tokens):
                # do NOT overwrite the full index — create a cropped view for the model input
                idx_cond = index[:, -block_size:]  # (B, T_cond) used for forward, keeps full 'index' intact

                # debug check before forward
                vocab_size = self.token_embedding_table.num_embeddings
                if idx_cond.min().item() < 0 or idx_cond.max().item() >= vocab_size:
                    raise IndexError(f"Pre-forward: index out of range: min={int(idx_cond.min())}, max={int(idx_cond.max())}, vocab={vocab_size}")

                logits, _ = self.forward(idx_cond)  # model forward on the cropped context
                logits = logits[:, -1, :]           # (B, vocab) -> last token's logits

                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                index_next = torch.multinomial(probs, num_samples=1)  # (B,1)

                if index_next.max().item() >= vocab_size or index_next.min().item() < 0:
                    print("index_next problematic:", index_next)
                    raise IndexError(f"Sampled index out of range: {index_next} vs vocab {vocab_size}")

                # append the sampled token to the FULL sequence
                index = torch.cat((index, index_next), dim=1)

        return index




print('Loading model parameters...')
with open('C:/Documents/model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model initialized")
m = model.to(device).eval() # put the model on GPU if available otherwise CPU


while True:
    prompt = input("Prompt:\n")
    if prompt.strip().lower() in ('exit', 'quit'):
        break
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    output = decode(m.generate(
                                context.unsqueeze(0),
                                max_new_tokens=500,
                                temperature=0.7
                                )[0].tolist()
                            )
    with open(chat_logs, 'a') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Prompt: {prompt}\n[{timestamp}] Output: {output}\n")
    print(f'Output:\n{output}')

import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse
import datetime
import os
import unicodedata
import psutil
import GPUtil

parser = argparse.ArgumentParser(description='')

# add an argument to the parser, specifying the expacted type and help message
parser.add_argument('-bs', type=str, required=True, help='Provide a batch_size')
parser.add_argument('-itr', type=str, required=True, help='Provide max iterations')
parser.add_argument('-evitr', type=str, required=True, help='Provide eval iterations')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu' # use cuda(gpu) if not available use cpu
print (device)

batch_size = int(args.bs)  # how many intigers can the model see at once
print(f'batch size: {args.bs}')
block_size = 512 # how many blocks can the model see at once
max_iters = int(args.itr)
print(f'max iterations: {args.itr}')
eval_iters = int(args.evitr)
print(f'eval iterations: {args.evitr}')
learning_rate = 1e-4 #3e-4, 3e-3, 1e-4, 1e-3
n_embd = 384 # creates a 384 attribute embedding_vector
n_layer = 4
n_head = 4
dropout = 0.2 # 20% neurons will be turned off to prevent overfitting

ram_usage_percent = psutil.virtual_memory().percent
gpu = GPUtil.getGPUs()[0]  # first GPU
gpu_usage_percent = gpu.load * 100  # convert from 0-1 to %

total_iterations = "log_files/total_interations.txt"
loss_logs = "log_files/loss_logs.txt"

chars = ""
with open ('vocab.txt', 'r', encoding='utf-8') as f: # use text and encode it with utf-8
        text = f.read() # let the script read the text
        chars = sorted(list(set(text))) # sort all used characters in text
    
vocab_size = len(chars) # sets the vocab_size to the number of characters in chars

SPECIAL_CHARS = {'\x00', '\x01', '\x02'}

def normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = ''.join(
        ch for ch in s
        if (unicodedata.category(ch)[0] != 'C') or (ch in SPECIAL_CHARS)
    )
    return s


string_to_int = { ch:i for i,ch in enumerate(chars) } # make string of characters into intigers (full numbers)
int_to_string = { i:ch for i,ch in enumerate(chars) } # make intigers into string of characters
unk_id = string_to_int.get("<unk>", 0)
encode = lambda s: [string_to_int.get(c, unk_id) for c in s] # encode the characters into the intigers
decode = lambda l: ''.join([int_to_string[i] for i in l]) # decode the intigers into characters

# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "C:/Documents/datasets/lamini/stage3_train.txt" if split == 'train' else "C:/Documents/datasets/lamini/stage3_val.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # determine the filew size and a random position to start reading
            file_size = len(mm)
            file_size = int(file_size)  # convert to int first
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # seek to the random pos and ead the block of text 
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # decode the block to a string ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r','')
            decoded_block = normalize_text(decoded_block)

            # train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data
 
def get_batch(split): # defines a function called get_batch that takes a single argument split, which tells us whether to use the training or validation data
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Each index i represents the start of a chunk of length block_size. We subtract block_size to ensure we don’t go out of bounds.
    #print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix]) # For each index i in ix, it slices a chunk from data of length block_size. Then stacks them into a tensor of shape (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1]for i in ix]) # This creates the target batch y, which is the same as x but shifted by one token to the right.The model will try to predict each next token in y given the corresponding token in x
    x, y = x.to(device), y.to(device)
    return x, y

# makes sure that torch doesnt use gradients at all so that will reduce computation, memoryusage just better for performance
@torch.no_grad() # build in pytorch decorator that disbles any gradient calculations
def estimate_loss():
    out = {}
    model.eval() # switches to evaluation mode, disables dropout
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # switches to training mode, enables dropout
    return out\

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
            for _ in range(max_new_tokens):
                # crop to last allowed context for pos embedding
                index = index[:, -self.position_embedding_table.num_embeddings:]

                # debug check before forward
                vocab_size = self.token_embedding_table.num_embeddings
                if index.min().item() < 0 or index.max().item() >= vocab_size:
                    raise IndexError(f"Pre-forward: index out of range: min={int(index.min())}, max={int(index.max())}, vocab={vocab_size}")

                logits, _ = self.forward(index)  # (B,T,vocab)
                logits = logits[:, -1, :]        # (B,vocab)

                # apply temperature
                logits = logits / temperature

                # convert to probabilities
                probs = F.softmax(logits, dim=-1)

                # sample next token
                index_next = torch.multinomial(probs, num_samples=1)  # (B,1)

                # safety clamp / check: multinomial should never return >=vocab_size
                if index_next.max().item() >= vocab_size or index_next.min().item() < 0:
                    print("index_next problematic:", index_next)
                    raise IndexError(f"Sampled index out of range: {index_next} vs vocab {vocab_size}")

                index = torch.cat((index, index_next), dim=1)

        return index


model = AsunaLanguageModel(vocab_size) # make the model and tell it how many tokens it can know
print('Loading model parameters...')
with open('C:/Documents/model-01-expanded-512.pkl', 'rb') as f:
   model = pickle.load(f)

print("Model initialized")
m = model.to(device) # put the model on GPU if available otherwise CPU

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # creates AdamW optimizer instance to update the models parameters
print("Starting training...")

try:
    for iter in range(max_iters): # starts a training loop that runs for the duration of max_iters each iteration performs an update step
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}, RAM usage: {ram_usage_percent}%, GPU usage: {gpu_usage_percent:.1f}%")
            with open(total_iterations, 'r') as f:
                total = int(f.read())
                total += eval_iters

            with open(total_iterations, 'w') as f:
                f.write(str(total))

            if iter == 0:
                with open(loss_logs, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"\n[{timestamp}] step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}\n")
            else:
                with open(loss_logs, 'a') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}\n")
        
        xb, yb = get_batch('train') # fetches a batch of input data (xb) and target labels (yb) 
        logits, loss = model.forward(xb, yb)  # passes the input and target batches thrugh the model, returns logits tyhe raw output and the loss scalar comparing the predictions and targets
        optimizer.zero_grad(set_to_none=True) # clears previos gradients from the model parameters
        loss.backward() # computes gradients of the loss with respect to all model parameters
        optimizer.step() # updates model parameters using the computed gradients and the optimizer algorithm
    print(loss.item()) # .item converts the tensor into a readable number
except KeyboardInterrupt:
    print('Training interrupted!')

# save the model 
with open('C:/Documents/model-01-expanded-512.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved')
print(f'New total iterations: {total}')

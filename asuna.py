import torch
import torch.nn as nn
from torch.nn import functional as F
import datetime
import os
import unicodedata
from transformers import GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu' # use cuda(gpu) if not available use cpu
print (device)

block_size = 512 # how many blocks can the model see at once
n_embd = 384 # creates a 384 attribute embedding_vector
n_layer = 12
n_head = 12
dropout = 0.2 # 20% neurons will be turned off to prevent overfitting 

tokenizer = GPT2Tokenizer.from_pretrained("tokenizer/") # load trained tokenizer from training
vocab_size = len(tokenizer) # len vocab size from the tokenizer lib
print("Tokenizer loaded")


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

    def generate(self, index, max_new_tokens, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.2):
        self.eval()
        with torch.no_grad():
            block_size = self.position_embedding_table.num_embeddings

            for _ in range(max_new_tokens):
                idx_cond = index[:, -block_size:]

                vocab_size = self.token_embedding_table.num_embeddings
                if idx_cond.min().item() < 0 or idx_cond.max().item() >= vocab_size:
                    raise IndexError(f"Pre-forward: index out of range")

                logits, _ = self.forward(idx_cond)
                logits = logits[:, -1, :]  # (B, vocab_size)

                # repetition penalty
                window = 128  # or block_size
                if repetition_penalty != 1.0:
                    for b in range(logits.size(0)):
                        recent = index[b, -window:].tolist()
                        seen_tokens = set(recent)
                        for token_id in seen_tokens:
                            if logits[b, token_id] > 0:
                                logits[b, token_id] /= repetition_penalty
                            else:
                                logits[b, token_id] *= repetition_penalty

                logits = logits / max(temperature, 1e-8)

                # top-k
                if top_k is not None:
                    values, _ = torch.topk(logits, top_k)
                    min_values = values[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_values, float('-inf'), logits)

                # top-p
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(probs, dim=-1)

                    cutoff = cum_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False

                    sorted_logits[cutoff] = float('-inf')
                    logits = torch.zeros_like(logits).scatter(
                        1, sorted_indices, sorted_logits
                    )

                probs = F.softmax(logits, dim=-1)
                index_next = torch.multinomial(probs, num_samples=1)

                if index_next.max().item() >= vocab_size or index_next.min().item() < 0:
                    raise IndexError(
                        f"Sampled index out of range: {index_next} vs vocab {vocab_size}"
                    )

                index = torch.cat((index, index_next), dim=1)

                if index_next.item() == tokenizer.eos_token_id:
                    break
            
        return index


print('Loading model parameters...')
model = AsunaLanguageModel(vocab_size)
model.load_state_dict(torch.load("C:/Documents/asuna.pt"))
model = model.to(device)
model.eval()
print("Model loaded")

# Special control tokens
USER_TOKEN = '\x01'
ASSISTANT_TOKEN = '\x02'

# Text normalization
def normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ") # replace unknown characters with ''
    s = ''.join(
        ch for ch in s
        if (unicodedata.category(ch)[0] != 'C') or (ch in {USER_TOKEN, ASSISTANT_TOKEN, '\x00'})
    )
    return s

chat_logs = "log_files/chat_logs.txt"
os.makedirs(os.path.dirname(chat_logs), exist_ok=True)

print("Chat interface ready. Type 'exit' or 'quit' to stop.\n")

while True:
    prompt = input("Prompt:\n")
    prompt = normalize_text(prompt)
    if prompt.strip().lower() in ('exit', 'quit'):
        break

    prompt = normalize_text(prompt)
    structured_prompt = f"{USER_TOKEN}{prompt}{ASSISTANT_TOKEN}"

    context_ids = tokenizer.encode(structured_prompt, add_special_tokens=False)
    context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    output_ids = model.generate(
        context,
        max_new_tokens=2500,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1
    )[0].tolist()

    # Extract only the generated response tokens (everything after the prompt)
    response_ids = output_ids[len(context_ids):]

    # Remove trailing EOS token if the model generated it
    eos_id = tokenizer.eos_token_id
    if response_ids and response_ids[-1] == eos_id:
        response_ids = response_ids[:-1]

    # Decode only the response, skipping any special tokens and cleaning up BPE artifacts
    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # logging
    with open(chat_logs, 'a', encoding='utf-8', errors='replace') as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] Prompt: {prompt}\n[{timestamp}] Output: {decoded}\n")

    print(f'Output:\n{decoded}')

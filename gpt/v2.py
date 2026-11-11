# Bigram LM synthesized from gpt-dev.ipynb

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -------------------------------------------------------

torch.manual_seed(1337)

# read and inspect the tinyshakespeare dataset
with open(os.path.join(os.path.dirname(__file__), 'input.txt'), 'r', encoding='utf-8') as f:
    text = f.read()
    
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores "affinities"
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # output of self-attention itself
        #out = self.proj(out)                                # linear projection, for residual
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    " a simple linear layer followed by a non-linearity "
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )
        
        
    def forward(self, x):
        return self.net(x)

# Block intersperses communication and computation
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = FeedForward(n_embd)                 # computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # note: layernorm applied before x goes into sa/ffwd
        x = x + self.sa(self.ln1(x))   # fork off (x +) to residual, do some communication and come back
        x = x + self.ffwd(self.ln2(x)) # fork off, do some computation, come back
        return x

# super simple bigram model (not anymore)
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        #self.blocks = nn.Sequential(
        #    Block(n_embd, n_head=4),
        #    Block(n_embd, n_head=4),
        #    Block(n_embd, n_head=4),
        #    nn.LayerNorm(n_embd),
        #)
        #self.sa_head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # ie. 4 heads of 8-dimensional self-attention
        #self.ffwd = FeedForward(n_embd) # with feed-forward on token level
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) - (Batch, Time, Channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) 
        x = tok_emb + pos_emb # (B,T,C) -- x holds not just token identities, but also the token positions within T
        
        #x = self.sa_head(x) # apply one head of self-attention. (B,T,C)  --- allows tokens to communicate with each other
        #x = self.sa_heads(x) # apply heads of self-attention. (B,T,C)    --- allows self-attention heads to communicate with each other
        #x = self.ffwd(x) # # apply single feed-forward layer. (B,T,C)    --- allows tokens time to "think on" what they found from each other
        x = self.blocks(x) # apply transformer blocks (B,T,C)            --- allows interspersing of communication+computation
        
        x = self.ln_f(x) # (B,T,vocab_size)                              --- apply layernorm
        logits = self.lm_head(x) # (B,T,vocab_size) -- decode
        
        if targets is None: 
            loss = None
        else:
            # reshape logits, targets because of how pytorch works (expects B*T,C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities 
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) # uses AdamW instead of SGD which makemore did

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    
    
    
# Single-head self-attention (loss ~2.5, 3000 iters):
""" 
ARI
HAThan, thivet'ly meirar ysuch, fo thest
K
CAV:
HEDIERESTAD Pfr:
BENENENLING CENNUCINA IO: yond houstoruloubleles, be sordweloe yot bed thed Ond Doue bou
His, to I lllat;
Kind yo Icad imoralceng, wpatsl my peer furs bust hasthe sor merals of ss whacer gaers, ver: biolllay seirost fat:
Thones som, wer ran
Toupls om?
Sll: hy athithos benciet bs fof.
Amathalcbenou liles caly, detave shpreearg
Whe wad wolf sd it totosthe, wank th in thit shin
S ht ande conit iul cert gpapam oun llkeavilint hmy r
"""

# Multi-head self-attention (4 heads) (loss ~2.27, 5000 iters):
"""
bachit yand thim his of intreysuch, fore wit
KEd Ve;
Anstend reardm: swhim whitl quelarsai thru.
I'
Thost thersblelts, be sornw-loe you bed'thed O.

DOUST:
Whe sefto I 
That ten, my thald hougarcencus parslers peer furs bust hatth'ssor me gest.

KASTh,
Thegacps, var: bis
Lupurs ifor in apter yes sont wer rand, sp.

LOHASll, hady btthos bencies be for.
ALIUCABCbe nos, you caly, detave shpr, mre
Whe wad wolf slill tot sthe, wanivelfill
KE be in
Shes and on: I witlecept gor am ounkieke thrins hatne
"""

# With feedforward on token level (loss ~2.24, 5000 iters):
"""
An your to drie'nd anden perth an, shy hon hee.
Thout shapet so I a to doinke Mude be hath youghdirme mod: lad thomis ond Wat, your weath

TUD:
Whyss stur; to sond, ound be thy
En arlie he semohe pine the of men. Willadell. Oncue
JORMOES:
Whe mar ford my nee!
Tas imh cone you the lot pothe ftior yousy thame thee
He falesp slesth, fand diker to ser's deanefike
Fer yady me.
Shist her eastith is your hom apelold Ris dike,
St-arde andint nord seve, to allwme.

Wheme, stramardy ming a of everendeme p
"""

# With transformer blocks interspersing communication+computation (3 blocks) (loss ~2.3, 5000 iters) -- model degrades:
"""
I sor beat trelincs Rromer
Fope, war on;
Bo wre thamernnchud colbele may hy thy to sithen.

GOD: Lall Merage, ase mranst biet ror, you buve shaickiso gat a thoom lof Jothies and hist mlat baith mancaland shhathime ther, luve Ce, bay.Ths soiy siliert orfour;
Domy:
I mote hery arag? Wou the dibelds?

EGENGG:
EaNGe bemnatsed of of sir.

HYORIN WI I hive'sl, as fary sit hour, ands the wen mist trer, asen
Af arvtereg ostositld of mere int of
ye hep of you twiml haloldmat seve of thy wheate so's ist t
"""

# With residual / skip connections (loss ~2.09, 5000 iters):
"""
NEDWAPER:
Thou suartst. HARD RI:
Ouc mell not scepell, my scome, hope,
Sre so sed had noble that, of thay say, o en bark that, Heart to faid I hol me: Gracled,---plains pry me a cutic zove you deef your sir:
Ten'd;
Wiling.

POMILIA:
Youthere,
Adwuth apurs of sayess, I dead che ke yeto ceveres a fore off now has; lorde to. Sintatt; man they your in his dise, make.
Stleat rester me agre astore mist ature the mauser my know if brayet! we fear our eave mads muster.

CATIAFOL:
A quares forsing the mo
"""

# With layernorm (loss ~2.07, 5000 iters):
"""
LUS:
Cere are to mave many.

GLOUCINTY YATA:
Trruntiow O-sis way, from 'd on of the frue come?

EDWARD Nomes is ands.
Athope interceself not scepela's pronobliees wis reast the had noble that
moldanges,
And I the so that, Heave to fain's too me: him, ke proplacess than was, wick,
Sit pulage again sike kencest winte.

Pwith chath, betut waw the purs of name am ke, musch''d  yeto kess:
Be, Bongery, not pos;
This he dien theses,
Hatith your in his do you fay; the his noter perthy
Falt a in as thou
"""

# With dropout and scaled up hyperparameters (loss ~, 5000 iters):
"""

"""

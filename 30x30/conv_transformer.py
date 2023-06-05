import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    """
    From "Attention is all you need" paper
    """
    def __init__(self, embed_dim=64, num_heads=2, dropout=0.0, att_dropout=0.1) -> None:
        super().__init__()
        head_dim = embed_dim//num_heads
        self.num_heads = num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(att_dropout)
    
    def forward(self, x, att_mask=None):
        batch_size, seq, embed_dim = x.size()
        q = self.q(x).reshape(batch_size, self.num_heads, seq, embed_dim//self.num_heads)
        k = self.k(x).reshape(batch_size, self.num_heads, seq, embed_dim//self.num_heads)
        v = self.v(x).reshape(batch_size, self.num_heads, seq, embed_dim//self.num_heads)

        # Gives output of batch_size x num_heads x (seq x seq)
        align = k@q.transpose(-2,-1)*self.scale

        # Assuming user only specifies mask in (seq x seq) dims, with -inf as mask value
        if att_mask is not None:
            if att_mask.size() != align.size()[-2:]:
                raise AssertionError("Mask has incorrect dimensions")
            att_mask = att_mask[None, None, :, :]
            # In-place transformation
            align.masked_fill_(att_mask, -float("inf"))

        att = F.softmax(align, dim=-1)
        att = self.att_dropout(att)
        self.dataload = (v.transpose(-2,-1)@att).reshape(batch_size, seq, embed_dim)
        return self.dropout(self.proj(self.dataload))


class ConvTransformerEncoderLayer(nn.Module):
    """
    Taken from "Attention is All you Need" paper
    """
    def __init__(self, d_model=64, 
                 nheads=2, 
                 dim_feedforward=256, 
                 dropout=0.0, attn_dropout=0.1, att_mask=None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.attn = Attention(d_model,nheads,dropout,attn_dropout)
        self.drop_1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(*[nn.Linear(d_model,dim_feedforward),nn.ReLU(),nn.Dropout(dropout),
                                  nn.LayerNorm(dim_feedforward), 
                                  nn.Linear(dim_feedforward,d_model),nn.Dropout(dropout)])
        self.ln2 = nn.LayerNorm(d_model)
        self.att_mask = att_mask

    def forward(self, x):
        attn_out = self.ln1(self.attn(x, self.att_mask)+x)
        return self.ln2(self.mlp(attn_out)+attn_out)

  
class RegressionTransformer(nn.Module):
    """
    Taken from "Attention is All You Need" paper, "Big Data Paradigm" paper
    """
    def __init__(self, d_model=64, nheads=2, seq_len=None,
                 num_encoder_layers=3, dim_feedforward=256, dropout=0.0, attn_dropout=0.1, mlp_ratio=4.0,
                 num_classes=1, seq_pool=True, pos_embed=False, att_mask=None) -> None:
        super().__init__()
        embedding_dim = d_model
        dim_feedforward = int(mlp_ratio*embedding_dim)
        if seq_len is None and pos_embed is True:
            raise ValueError('Sequence length not defined for positional embedding to be used')
        self.positional_emb = nn.Parameter(self.sinusoidal_embed(seq_len, embedding_dim),
                                        requires_grad=False) if pos_embed is True else None
        self.seq_pool = seq_pool
        # For seq_pooling
        self.importance_weight = nn.Linear(d_model, 1)

        self.in_dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.Sequential(*[ConvTransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout, attn_dropout, att_mask=att_mask)
                                             for _ in range(num_encoder_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model,num_classes)

    def forward(self, x):
        if self.positional_emb is not None:
            x += self.positional_emb
        norm_x = self.in_dropout(x)
        encode_x = self.encoder_blocks(norm_x)
        ln_x = self.out_ln(encode_x)
        if self.seq_pool:
            out = F.softmax(self.importance_weight(ln_x), dim=1).transpose(-2,-1)
            ln_x = (out@ln_x).squeeze(-2)
        else:
            ln_x = ln_x[:,0]

        return self.linear(ln_x)
    
    def sinusoidal_embed(self, seq, embed_dim):
        powers = torch.pow(1e+4, -2*torch.arange(embed_dim//2)/embed_dim)
        # Since we want embed_dim vectors over all time steps
        # we should broadcast sin and cos ops over all time steps
        # Expanding t for broadcasting, 
        t = torch.arange(seq)[:,None]
        pe = torch.zeros(seq,embed_dim)
        pe[:,0::2] = torch.sin(t*powers)
        pe[:,1::2] = torch.cos(t*powers)

        return pe
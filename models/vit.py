import numpy as np

import torch
import torch.nn as nn

# Positional embedding
def get_embeddings(seq_length, d):
    result = torch.ones(seq_length, d)
    for i in range(seq_length):
        for j in range(d):
            result[i][j] = np.sin(i/(10000**(j/d))) if j % 2 == 0 else np.cos(i/(10000**((j - 1)/d)))
    return result

# Multi-head self attention (MSA)
class MSA(nn.Module):
    def __init__(self, d, heads=4):
        super(MSA, self).__init__()
        self.d = d
        self.heads = heads

        assert d%heads == 0, f"Can't divide dimension {d} into {heads} heads"

        d_head = int(d/heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # sequences has shape (N, seq_length, token_dim)
        # we go into shape    (N, seq_length, n_heads, token_dim/n_heads)
        # and come back to    (N, seq_length, item_dim) (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head*self.d_head: (head + 1)*self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T/(self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# VIT block (with residual connection)
class VITBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4):
        super(VITBlock, self).__init__()
        self.hidden = hidden
        self.heads = heads

        self.norm1 = nn.LayerNorm(hidden)
        self.mhsa = MSA(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
                                 nn.Linear(hidden, mlp_ratio*hidden),
                                 nn.GELU(),
                                 #nn.Dropout(p=0.1),
                                 nn.Linear(mlp_ratio*hidden, hidden)
                                 #nn.Dropout(p=0.1)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class VIT(nn.Module):
    def __init__(self, 
                 patches=125,
                 patch_size=20, 
                 in_channels=1, 
                 blocks=2, 
                 heads=4,
                 hidden_d=256,
                 out_d=2,               # classes (roost/non-roost)  
                 verbose=False):        # to debug
        super(VIT, self).__init__()
        
        # Attributes
        self.patches = patches
        self.hidden_d = hidden_d
        self.blocks = blocks
        self.heads = heads
        self.verbose = verbose
        
        # Linear mapper
        self.linear_mapper = nn.Conv3d(1, self.hidden_d, patch_size, stride=patch_size)
 
        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 1D Embeddings (positional, temporal, and size)
        # self.pos_embed = nn.Parameter(torch.rand(self.patches + 1, self.hidden_d))
        
        # 3D positional encoding:
        # pos_embed_x = torch.tensor(get_embeddings(4, 85))
        # pos_embed_y = torch.tensor(get_embeddings(4, 85))
        # pos_embed_z = torch.tensor(get_embeddings(4, 86))
        # self.pos_embed = torch.zeros(self.patches + 1, self.hidden_d)
        # cont = 1
        # for i in range(4):
        #     for j in range(4):
        #         for k in range(4):
        #             self.pos_embed[cont,:] = torch.cat((pos_embed_x[i,:], pos_embed_y[j,:], pos_embed_z[k,:]), 0)
        #             cont += 1
        # self.pos_embed = nn.Parameter(self.pos_embed)

        # self.pos_embed.requires_grad = False
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([VITBlock(hidden_d, heads) for _ in range(blocks)])
        
        # Classifiation MLP
        self.mlp = nn.Sequential(
                                  nn.Linear(self.hidden_d, 1)
                                  #nn.Dropout(p=0.1)
                                  #nn.Softmax(dim=-1)
                                 )
        
    def forward(self, patches):
        
        N, C, H, W, D = patches.shape # D = C*H*W

        # [N,P,D] --> [N,P,hidden_d] map input patch to token
        tokens = self.linear_mapper(patches) # [1,256,5,5,5]
        _, P, h, w, d = tokens.shape
        tokens = tokens.view(N, 1, self.hidden_d, h*w*d) # [1,256,125]
        tokens = tokens.permute(0,1,3,2).squeeze() # [125,256]

        # [N,P,hidden_d] --> [N, patches+1, hidden_d] add class token
        out = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # if have positional embeddings
        # pos_embed = self.pos_embed.repeat(N, 1, 1)
        # out = out + pos_embed 
        # if self.verbose: print('after embeddings: %s'%str(out.shape))
        
        # Transformer blocks
        for block in self.blocks:
            out = block(out)
            if self.verbose: print('after blocks: %s'%str(out.shape))
        
        # Getting the classification token only
        out = out[:,0]
        out = self.mlp(out) # map to output dimension, output category distribution
        
        return out

# taken from https://github.com/tatp22/multidim-positional-encoding
class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc

# taken from https://github.com/tatp22/multidim-positional-encoding
class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

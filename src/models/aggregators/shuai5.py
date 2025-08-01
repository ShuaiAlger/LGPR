"""
    BoQ: A Place is Worth a Bag of learnable Queries (CVPR 2024)
    
    Paper: https://arxiv.org/abs/2405.07364
    GitHub repo: https://github.com/amaralibey/Bag-of-Queries
    
    Reference:
    @InProceedings{Ali-bey_2024_CVPR,
        author    = {Ali-bey, Amar and Chaib-draa, Brahim and Gigu\`ere, Philippe},
        title     = {{BoQ}: A Place is Worth a Bag of Learnable Queries},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        pages     = {17794-17803}
    }
    
"""


import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    










    
class BoQBlock(torch.nn.Module):
    """
    BoQ Block

    Args:
        in_dim (int): input dimension
        num_queries (int): number of queries to learn
        nheads (int): number of heads in the multihead attention. Defaults to 8.
    """
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        # self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)

        self.encoder = MambaVisionMixer(d_model=in_dim)

        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used to add context between the learned queries during training. 
        # you can cache their output in eval.
        # self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)

        self.mamba = MambaVisionMixer(d_model=in_dim)

        self.norm_q = torch.nn.LayerNorm(in_dim)
        ################
        
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        q = q + self.mamba(q)
        q = self.norm_q(q)
        
        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class Shuai5(torch.nn.Module):
    """
    Bag-of-Queries module

    Args:
        in_channels (int): Number of input channels (depth of input feature maps).
        proj_channels (int): Number of channels after the projection layer. Defaults to 512.
        num_queries (int): Number of queries to learn. Defaults to 32.
        num_layers (int): Number of BoQ blocks. Defaults to 2.
        row_dim (int): Row-wise projection dimension. Defaults to 32.
    """
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()
        
        # reduce input dimension using 3x3 conv
        self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        
        # normalize the input to the BoQ blocks
        self.norm_input = torch.nn.LayerNorm(proj_channels)
        
        # now the BoQ blocks input dimension is proj_channels
        boq_in_dim = proj_channels
        
        # create the BoQ blocks (each head of the self-attention has a dimension of 64)
        self.boqs = torch.nn.ModuleList([
            BoQBlock(boq_in_dim, num_queries, nheads=boq_in_dim//64) for _ in range(num_layers)])
        
        # the outputs of all BoQ blocks are concatenated and projected to row_dim
        self.fc = torch.nn.Linear(num_layers*num_queries, row_dim)
        
    def forward(self, x):
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns













import torch as th
import torch.nn as nn
import math
import numpy
from utils.decorators import *
from rotary_embedding_torch import RotaryEmbedding
from mamba_ssm import Mamba


    
class ConvBlock(nn.Module):
    def __init__(self, d_conv, dilation, online=False):
        super().__init__()
        padding = (1,2*dilation) if online else (1, dilation)
        self.conv = nn.Sequential(nn.Conv2d(d_conv, d_conv*2, (3,3), padding=padding, dilation=(1,dilation)),
                                  nn.InstanceNorm2d(d_conv*2, affine=True),
                                  nn.GLU(dim=-3),
                                  )

        self.online = online
        self.dilation = dilation

    def forward(self, x):

        y = self.conv(x)
        return  x + y if not self.online else x + y[...,:-self.dilation]
            
            
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        if rope: self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        if hasattr(self, 'rope'):
            query, key = self.apply_rope(query, key)
        
        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with th.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @th.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key
    


class LinformerAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
        selfattn=True,
        emb_dim_kv=None,
        Ekv=None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout


        if rope: self.rope = rope
        
        self.selfattn=selfattn
        if selfattn:
            self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        else:
            self.kv = nn.Linear(emb_dim_kv, attention_dim * 2, bias=False)
            self.q = nn.Linear(emb_dim, attention_dim, bias=False)
            
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))
        # Linformer Projection
        self.Ekv = Ekv
       
        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input, kv=None):
        if not self.selfattn:
            assert kv is not None, "In cross-attn, kv input is required."

        # get query, key, and value
        query, key, value = self.get_qkv(input, kv)
            
        # rotary positional encoding
        if hasattr(self, 'rope'):
            query, key = self.apply_rope(query, key)
        
        # Linformer Projection
        key, value = self.Ekv(key, value)
        
        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with th.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, query, kv):
        B, Lq, _ = query.shape

        if self.selfattn:
            x = self.qkv(query)
            x = x.view(B, Lq, 3, self.n_heads, -1).permute(2,0,3,1,4)
            q, k, v = x[0], x[1], x[2]
        else:
            q = self.q(query).view(B, Lq, self.n_heads, -1).permute(0,2,1,3)
            B, Lk, _ = kv.shape
            kv_proj = self.kv(kv).view(B, Lk, 2, self.n_heads, -1).permute(2,0,3,1,4)
            k, v = kv_proj[0], kv_proj[1]
        return q, k, v

    @th.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key
    

    
    
class MHSA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout_rate: float, rope):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.block = MultiHeadSelfAttention(d_model, d_model, n_head, dropout_rate, rope)
    
    def forward(self, x: th.Tensor):
        """
        Compute encoded features.
            :param th.Tensor x: encoded source features (batch, max_time_in, size)
            :param th.Tensor mask: mask for x (batch, max_time_in)
            :rtype: Tuple[th.Tensor, th.Tensor]
        """
        y = self.layer_norm(x)
        y = self.block(y)
        
        return x + y


class LinMHSA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout_rate: float, Ekv):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.block = LinformerAttention(d_model, d_model, n_head, dropout_rate, Ekv=Ekv)
    
    def forward(self, x: th.Tensor):
        # x: B*T, F, C
        y = self.layer_norm(x)
        y = self.block(y)
        return x + y


class LinMHCA(nn.Module):
    def __init__(self, d_model: int, d_model_kv:int, n_head: int, dropout_rate: float, Ekv):
        super().__init__()
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.block = LinformerAttention(d_model, d_model, n_head, dropout_rate, selfattn=False, emb_dim_kv=d_model_kv, Ekv=Ekv)
        
    def forward(self, x: th.Tensor, kv: th.Tensor):
        # x: B*T, F(orig + pad), C
        len_seq = kv.shape[1]
        orig, pad = x[:,:len_seq], x[:,len_seq:]
        if len_seq < x.shape[1]:
            q = self.layer_norm_q(pad)
            pad_updated = self.block(q, kv)
            return th.cat([orig, pad + pad_updated], dim=1)
        else:
            return x

      

class ConvFFN(th.nn.Module):
    def __init__(self, d_model: int, d_hidden:int, kernel_size: int, dilation: int, dropout_rate: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.net1 = nn.Conv1d(d_model, 2*d_hidden, kernel_size, dilation=dilation, padding=((kernel_size-1)*dilation)//2)
        self.net2 = nn.Conv1d(d_hidden, d_model, kernel_size, padding=(kernel_size-1)//2)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        y = self.layernorm(x)
        y = y.permute(0, 2, 1)
        y = self.net1(y)
        y = self.silu(y[:,:y.shape[1]//2])*y[:,y.shape[1]//2:]
        y = self.net2(y)
        y = y.permute(0, 2, 1)
        return x + 0.5*y
    


    
class TransEncoder(nn.Module):
    def __init__(self, d_model: int, d_hidden:int, n_head: int, kernel_size: int, dropout_rate: float, rope):
        super().__init__()
        self.ffn_1 = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate) 
        self.sa = MHSA(d_model, n_head, dropout_rate, rope=rope)
        self.ffn_2 = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate) 

    def forward(self, x: th.tensor):
        #* B, L, C
        x = self.ffn_1(x)
        x = self.sa(x)
        x = self.ffn_2(x)
        return x

class LinTransEncoder(nn.Module):
    def __init__(self, d_model: int, d_hidden:int, n_head: int, kernel_size: int, dropout_rate: float, Ekv):
        super().__init__()
        self.ffn_1 = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate) 
        self.sa = LinMHSA(d_model, n_head, dropout_rate, Ekv)
        self.ffn_2 = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate) 

    def forward(self, x: th.tensor):
        #* B, L, C
        x = self.ffn_1(x)
        x = self.sa(x)
        x = self.ffn_2(x)
        return x

class TransDecoder(nn.Module):
    def __init__(self, d_model: int, d_hidden:int, n_head: int, kernel_size: int, dropout_rate: float, rope):
        super().__init__()
        self.ffn1 = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate)
        self.sa = MHSA(d_model, n_head, dropout_rate, rope=rope)
        self.ffn2 = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate)

    def forward(self, x: th.tensor):
        #* B, L, C
        x = self.ffn1(x)
        x = self.sa(x)
        x = self.ffn2(x)
        return x

class LinTransDecoder(nn.Module):
    def __init__(self, d_model: int, d_model_kv:int, d_hidden:int, n_head: int, kernel_size: int, dropout_rate: float, Ekv):
        super().__init__()
        self.ca = LinMHCA(d_model, d_model_kv, n_head, dropout_rate, Ekv)
        self.sa = LinMHSA(d_model, n_head, dropout_rate, Ekv)
        self.ffn = ConvFFN(d_model, d_hidden, kernel_size, 1, dropout_rate)

    def forward(self, x: th.tensor, kv: th.tensor):
        #* B, L, C
        x = self.ca(x, kv)
        x = self.sa(x)
        x = self.ffn(x)
        return x


class MambaV1Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout_rate: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.mamba_fw = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: th.Tensor):

        y = self.layernorm(x)
        y = self.mamba_fw(y)

        return self.dropout(y) + x

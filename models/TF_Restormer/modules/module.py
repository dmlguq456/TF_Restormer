import sys
sys.path.append('../')

import torch
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn


from utils.decorators import *
from .network import *




class F_Linear(nn.Module):
    def __init__(
        self,
        seq_len,
        proj_len,
        n_heads,
        kv_shared=True,
        ):
        super().__init__()
        
        # Linformer Projection
        self.kv_shared = kv_shared
        if kv_shared:
            self.Ekv1 = nn.Parameter(th.empty(n_heads, proj_len, seq_len)) 
            nn.init.xavier_uniform_(self.Ekv1, gain=0.01)
        else:
            self.Ek1 = nn.Parameter(th.empty(n_heads, proj_len, seq_len)) 
            self.Ev1 = nn.Parameter(th.empty(n_heads, proj_len, seq_len))
            for p in (self.Ek1, self.Ev1): nn.init.xavier_uniform_(p, gain=0.01)

    def forward(self, key, value):
        valid_len = key.shape[-2]
        if self.kv_shared:
            key   = th.einsum('hln, bhnd -> bhld', self.Ekv1[...,:valid_len], key)
            value = th.einsum('hln, bhnd -> bhld', self.Ekv1[...,:valid_len], value)
        else:
            key   = th.einsum('hln, bhnd -> bhld', self.Ek1[...,:valid_len], key)
            value = th.einsum('hln, bhnd -> bhld', self.Ev1[...,:valid_len], value)

        return key, value
        
    
class Encoder(nn.Module):
    def __init__(self,  online: bool, d_model: int, d_freq: int, freq_pe: bool):
        super().__init__()

        class FreqPositionalEncoding(nn.Module):
            def __init__(self, d_freq, d_model):
                super().__init__()
                self.p = nn.Parameter(th.zeros(1, d_freq, 1, d_model), requires_grad=True)

            def forward(self, x):
                return x + self.p[:,:x.shape[1]]

        self.embed = nn.Conv2d(2, d_model, (3,3), padding=(1,1))
        self.online = online
        self.layernorm = nn.LayerNorm(d_model)
        if freq_pe:
            self.freq_pe = FreqPositionalEncoding(d_freq, d_model)

    def forward(self, x: torch.tensor):
        # x : B, F, T, 2
        x_r = x[...,0] # B, F, T
        x_i = x[...,1] # B, F, T
        xs = torch.stack([x_r, x_i], dim=1) # B, 3, F, T
        xs = self.embed(xs) # B, C, F, T
        # if self.online:
        #     xs = xs[...,:-2]
        xs = xs.permute(0, 2, 3, 1).contiguous() # B, F, T, C
        xs = self.layernorm(xs)
        if hasattr(self, 'freq_pe'):
            xs = self.freq_pe(xs)

        return xs

    
class TF_Encoder(nn.Module):

    def __init__(self,  online: bool, time_module: dict, freq_module: dict, Ekv, rope):
        super().__init__()

        class FreqModule(nn.Module):
            def __init__(self, d_model: int, d_hidden:int, n_head: int, 
                         kernel_size: int, dropout_rate: float, Ekv):
                super().__init__()
                
                self.block = LinTransEncoder(d_model, d_hidden, n_head, kernel_size, dropout_rate, Ekv)

            def forward(self, x: torch.tensor):
                B, F, T, C = x.shape
                x = x.permute(0, 2, 1, 3).reshape(B*T, F, C) # B, T, F, C
                x = self.block(x)
                x = x.reshape(B, T, F, C).permute(0, 2, 1, 3) # B, T, F, C
                
                return x

        class OfflineTimeModule(nn.Module):
            def __init__(self, d_model: int, d_hidden:int, n_head: int, kernel_size: int, dropout_rate: float, rope):
                super().__init__()

                self.block = TransEncoder(d_model, d_hidden, n_head, kernel_size, dropout_rate, rope)

            def forward(self, x: torch.tensor):
                B, F, T, C = x.shape
                x = x.reshape(B*F, T, C)
                x = self.block(x)
                x = x.reshape(B, F, T, C)
                
                return x

        class OnlineTimeModule(nn.Module):
            def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout_rate: float):
                super().__init__()

                self.block = nn.Sequential(*[MambaV1Block(d_model, d_state, d_conv, expand, dropout_rate) for _ in range(2)])

            def forward(self, x: torch.tensor):
                B, F, T, C = x.shape
                x = x.reshape(B*F, T, C)
                x = self.block(x)
                x = x.reshape(B, F, T, C)
                
                return x

        self.frame_wise_block = FreqModule(**freq_module, Ekv=Ekv)
        if online:
            self.freq_wise_block = OnlineTimeModule(**time_module['online'])
        else:
            self.freq_wise_block = OfflineTimeModule(**time_module['offline'], rope=rope) 


    def forward(self, x: torch.tensor):
        x = self.frame_wise_block(x)
        x = self.freq_wise_block(x)
        
        return x


class TF_Decoder(nn.Module):

    def __init__(self, online: bool, time_module: dict, freq_module: dict, Ekv, rope):
        super().__init__()

        class FreqModule(nn.Module):
            def __init__(self, d_model: int, d_model_kv:int, d_hidden:int, n_head: int, 
                         kernel_size: int, dropout_rate: float, Ekv):
                super().__init__()
                
                self.block = LinTransDecoder(d_model, d_model_kv, d_hidden, n_head, kernel_size, dropout_rate, Ekv)

            def forward(self, x: torch.tensor, kv: torch.tensor, pad_len: int):
                B, F, T, C = x.shape
                _, F_kv, _, C_kv = kv.shape
                x = x.permute(0, 2, 1, 3) #* B, T, F, C
                kv = kv.permute(0, 2, 1, 3) #* B, T, F, C
                x = x.reshape(B*T, F, C)
                kv = kv.reshape(B*T, F_kv, C_kv)
                x = self.block(x, kv)
                x = x.reshape(B, T, F, C)
                x = x.permute(0, 2, 1, 3) #* B, T, F, C
                
                return x

        class OfflineTimeModule(nn.Module):
            def __init__(self, d_model: int, d_hidden:int, n_head: int, kernel_size: int, dropout_rate: float, rope):
                super().__init__()

                self.block = TransEncoder(d_model, d_hidden, n_head, kernel_size, dropout_rate, rope)

            def forward(self, x: torch.tensor):
                B, F, T, C = x.shape
                x = x.reshape(B*F, T, C)
                x = self.block(x)
                x = x.reshape(B, F, T, C)
                
                return x

        class OnlineTimeModule(nn.Module):
            def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout_rate: float):
                super().__init__()

                self.block = nn.Sequential(*[MambaV1Block(d_model, d_state, d_conv, expand, dropout_rate) for _ in range(2)])

            def forward(self, x: torch.tensor):
                B, F, T, C = x.shape
                x = x.reshape(B*F, T, C)
                x = self.block(x)
                x = x.reshape(B, F, T, C)
                
                return x

        self.frame_wise_block = FreqModule(**freq_module, Ekv=Ekv)
        if online:
            self.freq_wise_block = OnlineTimeModule(**time_module['online'])
        else:
            self.freq_wise_block = OfflineTimeModule(**time_module['offline'], rope=rope) 


    def forward(self, x: torch.tensor, kv: torch.tensor, pad_len: int):
        x = self.frame_wise_block(x, kv, pad_len)
        x = self.freq_wise_block(x)
        
        return x


class FreqUpsampleToken(nn.Module):
    def __init__(self, d_model: int, d_model_out: int, d_freq_min: int, d_freq_max: int):
        super().__init__()
        # compute target frequency bins and padding size
        pad_size = d_freq_max - d_freq_min
        self.proj = nn.Linear(d_model, d_model_out)
        self.mask_token = nn.Parameter(torch.zeros(1, pad_size, 1, d_model_out))
        nn.init.xavier_uniform_(self.mask_token)
        self.layernorm = nn.LayerNorm(d_model_out)
        self.max_F = d_freq_max
        self.min_F = d_freq_min

    def forward(self, x: torch.Tensor, out_F=769) -> torch.Tensor:
        """
        x: [B, F, T, C_in]
        returns: [B, target_F, T, C_out]
        """
        x = self.proj(x)
        B, F, T, C = x.shape
        out_F = min(out_F, self.max_F)
        if out_F < self.min_F:
            raise ValueError(f"min_F ({out_F}) should be greater than or equal to target freq. bins ({self.min_F})")
        
        pad_size = out_F - F #! pad only when out_F > F
        if pad_size > 0:
            offset = F - self.min_F
            pad_ = self.mask_token[:,offset:offset+pad_size]
            pad_ = pad_.expand(B, pad_size, T, C)
            x = torch.cat([x, pad_], dim=1)
        x = self.layernorm(x)
        return x, pad_size


class Decoder(nn.Module):
    def __init__(self, online: bool, d_model):
        super().__init__()


        class MagCompEstim(nn.Module):
            def __init__(self, online, d_model):
                super().__init__()
                self.out = nn.Sequential(nn.Conv2d(d_model, 1, (3,3), padding=(1,1)),
                                         nn.Softplus())
                self.online = online

            def forward(self, x):
                x = self.out(x)
                return x

        class ComplexCompEstim(nn.Module):
            def __init__(self, online, d_model):
                super().__init__()
                self.out = nn.Sequential(nn.Conv2d(d_model, 2, (3,3), padding=(1,1)), 
                                         nn.Tanh())
                self.online = online

            def forward(self, x):
                x = self.out(x)
                return x

        self.mag = MagCompEstim(online, d_model)
        self.phase = ComplexCompEstim(online, d_model)

    def forward(self, x):
        # x : B, F, T, C
        B, F, T, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, F, T
        y_mag = self.mag(x) # B, 1, F, T
        y_complex = self.phase(x) # B, 1, F, T
        comp = y_mag*y_complex
        comp = comp.permute(0, 2, 3, 1) # B, F, T, 2

        return comp



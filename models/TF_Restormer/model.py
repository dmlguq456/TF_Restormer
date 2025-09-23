import sys
sys.path.append('../')

import torch
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# import sys
# sys.path.append('/home/Uihyeop/nas_Uihyeop/NN_SE/TPMCN_CSS')
# from models.TPMCN_tiny_RM_v1.modules.module import *

from utils.decorators import *
from .modules.module import *

STAGE_BLOCK = {"Encoder": TF_Encoder, "Decoder": TF_Decoder}

# @logger_wraps()
class Model_Enhance(torch.nn.Module):
    def __init__(self,
                 online: bool,
                 input_embedding: dict,
                 freq_linear: dict,
                 encoder_stage: dict,
                 freq_upsampler: dict,
                 decoder_stage: dict,
                 output_spec:dict):
        super().__init__()

        class TF_stage(torch.nn.Module):
            def __init__(self, block_type: str, RoPE: dict, TF_block_Stage: dict, num_repeat: int, Ekv):
                super().__init__()
                rope = RotaryEmbedding(RoPE['d_model'] // RoPE['n_head'], theta=RoPE['theta'])
                self.tf_block = torch.nn.ModuleList(
                        [STAGE_BLOCK[block_type](**TF_block_Stage, Ekv=Ekv, rope=rope) for _ in range(num_repeat)])
                self.layer_norm = nn.LayerNorm(RoPE["d_model"])
                    
            def forward(self, x, kv=None, pad_len=None):
                for block in self.tf_block:
                    x = block(x, kv, pad_len) if kv is not None else block(x)
                x = self.layer_norm(x)
                return x


        self.online = online

        Ekv = F_Linear(**freq_linear)

        self.input_embed = Encoder(**input_embedding)
        self.encoder = TF_stage(**encoder_stage, Ekv=Ekv)

        self.up = FreqUpsampleToken(**freq_upsampler)
        self.decoder = TF_stage(**decoder_stage, Ekv=Ekv)
        self.estimator = Decoder(**output_spec)

        
    def forward(self, x, out_F=961):
        # x : (B), F, T, 2
        if len(x.shape) == 3: # When No Batch Dimension
            x = x.unsqueeze(0)
            
        if x.shape[1] > out_F:
            x = x[:, :out_F]  #! Truncate to out_F frequency bins
            
        if not self.online:
            x, x_scale = self.norm(x)

        # encoder
        x_enc = self.input_embed(x)
        B, F, T, C = x_enc.shape
        pos_f = self.sinusoids(F, C)
        pos_f = pos_f.reshape(1, F, 1, C).to(x_enc.device)
        x_enc = x_enc + pos_f
        x_enc = self.encoder(x_enc)
            
        # upto 16k decoder
        x_dec, pad_len = self.up(x_enc, out_F)
        x_dec = self.decoder(x_dec, x_enc, pad_len)

        # Output
        y = self.estimator(x_dec)

        if not self.online:
            y = self.inorm(y, x_scale)
            
        return y # B, out_F, T, 2

    
    def norm(self, x, Xscale=None):
        if Xscale == None:
            Xabs = torch.sqrt(x[...,0]**2 + x[...,1]**2) # B, T, F
            Xscale = Xabs.mean(dim=(1,2), keepdims=True) + 1.0e-8 # B, 1, 1
            Xscale = Xscale.unsqueeze(-1)
        return x / Xscale, Xscale
        
    def inorm(self, x, Xscale):
        return x * Xscale


    def sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)    

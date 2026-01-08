import torch
import torch.nn as nn
import math
from utils.utils import QuantLinear
from transformers.pytorch_utils import Conv1D

_CURRENT_QUANT_CONFIG = None

@staticmethod
def set_active_quant_config(precision: str):
    global _CURRENT_QUANT_CONFIG
    _CURRENT_QUANT_CONFIG = precision

@staticmethod
def get_active_quant_config():
    return _CURRENT_QUANT_CONFIG

class LoRALayer(nn.Module):
    def __init__(self, layer, precisions: list, r: int = 4, alpha: float = 1.0, ):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.r = r
        self.precisions = precisions 

        if isinstance(layer, (Conv1D, QuantLinear)):
            in_features = layer.weight.shape[0]
            out_features = layer.weight.shape[1]
        else:
            in_features = layer.in_features
            out_features = layer.out_features
        
        self.lora_A = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(in_features, r)) for p in precisions
        })
        self.lora_B = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, out_features)) for p in precisions
        })
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.precisions:
            nn.init.kaiming_uniform_(self.lora_A[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[p])
    
    def forward(self, x):
        precision = get_active_quant_config()
        out = self.layer(x)
        if precision is not None and precision in self.precisions:
            lora = (x @ self.lora_A[precision]) @ self.lora_B[precision]
            out = out + self.alpha * lora

        else:
            raise ValueError(f"Precision {precision} not in {self.precisions}")
       
        return out


class LoraAttentionKV(nn.Module):
    def __init__(self, layer, precisions: list, r: int = 4, alpha: float = 1.0, hidden_dim: int = 768):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.r = r
        self.hidden_dim = hidden_dim
        self.precisions = precisions
        
        self.lora_A_k = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(hidden_dim, r)) for p in precisions
        })
        self.lora_B_k = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, hidden_dim)) for p in precisions
        })
        
        self.lora_A_v = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(hidden_dim, r)) for p in precisions
        })
        self.lora_B_v = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, hidden_dim)) for p in precisions
        })
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for p in self.precisions:
            nn.init.kaiming_uniform_(self.lora_A_k[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[p])
            nn.init.kaiming_uniform_(self.lora_A_v[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v[p])
    
    def forward(self, x):
        precision = get_active_quant_config()
        base_out = self.layer(x)
        
        if precision is not None and precision in self.precisions:
            lora_k = (x @ self.lora_A_k[precision]) @ self.lora_B_k[precision]
            lora_v = (x @ self.lora_A_v[precision]) @ self.lora_B_v[precision]
            zeros_q = torch.zeros_like(lora_k)
            lora_out = torch.cat([zeros_q, lora_k, lora_v], dim=-1)
            base_out = base_out + self.alpha * lora_out

        else:
            raise ValueError(f"Precision {precision} not in {self.precisions}")
        
        return base_out


def apply_lora_to_model(model, precisions: list, r: int = 4, alpha: float = 1.0):
    """
    Wrap c_attn layers with LoraAttentionKV and other QuantLinear layers with LoRALayer.
    
    Args:
        model: The GPT2 model to modify
        precisions: List of precision keys for LoRA adapters (e.g., ["4bit", "8bit"])
        r: LoRA rank
        alpha: LoRA scaling factor
    """

    for name, module in list(model.named_modules()):
        if not isinstance(module, QuantLinear):
            continue
        
        # Get parent module and attribute name
        name_parts = name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        attr_name = name_parts[-1]
        
        # Wrap c_attn with LoraAttentionKV, others with LoRALayer
        if attr_name == 'c_attn':
            hidden_dim = module.weight.shape[0]  # input features
            wrapped = LoraAttentionKV(module, precisions, r=r, alpha=alpha, hidden_dim=hidden_dim)
        else:
            wrapped = LoRALayer(module, precisions, r=r, alpha=alpha)
        
        setattr(parent, attr_name, wrapped)
    
    print(get_active_quant_config())
    return model

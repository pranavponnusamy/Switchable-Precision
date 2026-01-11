import torch
import torch.nn as nn
import math
from utils.utils import QuantLinear
from transformers.pytorch_utils import Conv1D

_CURRENT_QUANT_CONFIG = None

def set_active_quant_config(precision: str):
    """Set the currently active quantization precision config name."""
    global _CURRENT_QUANT_CONFIG
    _CURRENT_QUANT_CONFIG = precision

def get_active_quant_config():
    """Get the currently active quantization precision config name."""
    return _CURRENT_QUANT_CONFIG

class LoRALayer(nn.Module):
    def __init__(self, layer, precisions: list, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.r = r
        self.precisions = precisions
        self.scaling = alpha / r  # Standard LoRA scaling

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
            out = out + self.scaling * lora
        else:
            raise ValueError(f"Precision {precision} not in {self.precisions}")
       
        return out


class LoraAttentionQKV(nn.Module):
    """LoRA adapter for attention c_attn that adapts Q, K, and V projections."""
    def __init__(self, layer, precisions: list, r: int = 4, alpha: float = 1.0, hidden_dim: int = 768):
        super().__init__()
        self.layer = layer
        self.alpha = alpha
        self.r = r
        self.hidden_dim = hidden_dim
        self.precisions = precisions
        self.scaling = alpha / r  # Standard LoRA scaling
        
        # LoRA adapters for Q, K, V
        self.lora_A_q = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(hidden_dim, r)) for p in precisions
        })
        self.lora_B_q = nn.ParameterDict({
            p: nn.Parameter(torch.zeros(r, hidden_dim)) for p in precisions
        })
        
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
            nn.init.kaiming_uniform_(self.lora_A_q[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_q[p])
            nn.init.kaiming_uniform_(self.lora_A_k[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[p])
            nn.init.kaiming_uniform_(self.lora_A_v[p], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_v[p])
    
    def forward(self, x):
        precision = get_active_quant_config()
        base_out = self.layer(x)
        
        if precision is not None and precision in self.precisions:
            lora_q = (x @ self.lora_A_q[precision]) @ self.lora_B_q[precision]
            lora_k = (x @ self.lora_A_k[precision]) @ self.lora_B_k[precision]
            lora_v = (x @ self.lora_A_v[precision]) @ self.lora_B_v[precision]
            lora_out = torch.cat([lora_q, lora_k, lora_v], dim=-1)
            base_out = base_out + self.scaling * lora_out
        else:
            raise ValueError(f"Precision {precision} not in {self.precisions}")
        
        return base_out


# Keep old class for backwards compatibility
LoraAttentionKV = LoraAttentionQKV


def apply_lora_to_model(model, precisions: list, r: int = 4, alpha: float = 1.0, lora_attention: bool = False, lora_mlp: bool = True):
    """
    Wrap QuantLinear layers with LoRA adapters.
    
    - c_attn: Wrapped with LoraAttentionQKV (adapts Q, K, V)
    - c_proj and c_fc: Wrapped with LoRALayer
    
    Args:
        model: The GPT2 model to modify
        precisions: List of precision keys for LoRA adapters (e.g., ["4bit", "8bit"])
        r: LoRA rank
        alpha: LoRA scaling factor (effective scaling is alpha/r)
    """

    for name, module in list(model.named_modules()):
        if not isinstance(module, QuantLinear):
            continue
        
        # Get parent module and attribute name
        name_parts = name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        layer_type = name_parts[-2]
        attr_name = name_parts[-1]

        # Only apply LoRA based on lora_attention or lora_mlp flags
        if attr_name == 'c_attn' and lora_attention:
            hidden_dim = module.weight.shape[0]  # input features
            wrapped = LoraAttentionQKV(module, precisions, r=r, alpha=alpha, hidden_dim=hidden_dim)
        elif lora_attention and layer_type == 'attention':
            wrapped = LoRALayer(module, precisions, r=r, alpha=alpha)

        elif layer_type == 'mlp' and attr_name in ['c_proj', 'c_fc'] and lora_mlp:
            wrapped = LoRALayer(module, precisions, r=r, alpha=alpha)
        else:
            print(f"Skipping {name_parts} because it is not an attention or MLP layer", flush=True)
            continue
        setattr(parent, attr_name, wrapped)
    
    print(f"Applied LoRA with r={r}, alpha={alpha} (scaling={alpha/r})")
    print(f"Active config: {get_active_quant_config()}")
    return model

def save_lora(model, path, precisions: list):
    """
    Save the LoRA parameters of the model, organized by precision.
    
    Saved format: {precision1: {weights}, precision2: {weights}, ...}
    """
    lora_state_dict = {}
    
    for precision in precisions:
        precision_weights = {}
        for name, param in model.state_dict().items():
            if 'lora_' in name and precision in name:
                precision_weights[name] = param.clone()
        lora_state_dict[precision] = precision_weights
        print(f"Collected {len(precision_weights)} params for {precision}")
    
    torch.save(lora_state_dict, path)
    print(f"LoRA model saved to {path}")


def _add_precision_to_layer(module, precision: str, weights: dict, module_prefix: str):
    """
    Add a new precision to a LoRA layer's ParameterDicts and precisions list.
    """
    if isinstance(module, LoRALayer):
        # Get dimensions from existing params
        existing_prec = module.precisions[0]
        in_features, r = module.lora_A[existing_prec].shape
        _, out_features = module.lora_B[existing_prec].shape
        
        # Add new precision to the list
        if precision not in module.precisions:
            module.precisions.append(precision)
        
        # Create new parameters in the ParameterDicts
        a_key = f"{module_prefix}.lora_A.{precision}"
        b_key = f"{module_prefix}.lora_B.{precision}"
        
        if a_key in weights:
            module.lora_A[precision] = nn.Parameter(weights[a_key].clone())
        else:
            module.lora_A[precision] = nn.Parameter(torch.zeros(in_features, r))
            
        if b_key in weights:
            module.lora_B[precision] = nn.Parameter(weights[b_key].clone())
        else:
            module.lora_B[precision] = nn.Parameter(torch.zeros(r, out_features))
            
    elif isinstance(module, LoraAttentionQKV):
        # Get dimensions from existing params
        existing_prec = module.precisions[0]
        hidden_dim, r = module.lora_A_q[existing_prec].shape
        
        # Add new precision to the list
        if precision not in module.precisions:
            module.precisions.append(precision)
        
        # Load Q, K, V LoRA params
        for suffix in ['q', 'k', 'v']:
            lora_A = getattr(module, f'lora_A_{suffix}')
            lora_B = getattr(module, f'lora_B_{suffix}')
            
            a_key = f"{module_prefix}.lora_A_{suffix}.{precision}"
            b_key = f"{module_prefix}.lora_B_{suffix}.{precision}"
            
            if a_key in weights:
                lora_A[precision] = nn.Parameter(weights[a_key].clone())
            else:
                lora_A[precision] = nn.Parameter(torch.zeros(hidden_dim, r))
                
            if b_key in weights:
                lora_B[precision] = nn.Parameter(weights[b_key].clone())
            else:
                lora_B[precision] = nn.Parameter(torch.zeros(r, hidden_dim))


def load_lora(model, path, precision: str = None):
    lora_state_dict = torch.load(path, weights_only=True)
    
    # Determine which precisions to load
    if precision is not None:
        if precision not in lora_state_dict:
            available = list(lora_state_dict.keys())
            raise ValueError(f"Precision '{precision}' not found. Available: {available}")
        precisions_to_load = [precision]
    else:
        precisions_to_load = list(lora_state_dict.keys())
    
    # Find all LoRA layers and add precisions dynamically
    for prec in precisions_to_load:
        weights = lora_state_dict[prec]
        
        for name, module in model.named_modules():
            if isinstance(module, (LoRALayer, LoraAttentionQKV)):
                # Check if this precision already exists
                if prec not in module.precisions:
                    _add_precision_to_layer(module, prec, weights, name)
                    print(f"Added precision '{prec}' to {name}")
                else:
                    # Precision exists, just load the weights
                    model.load_state_dict(weights, strict=False)
        
        print(f"Loaded LoRA params for precision '{prec}'")
    
    return precisions_to_load

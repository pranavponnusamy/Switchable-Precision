from dataclasses import dataclass
from transformers.pytorch_utils import Conv1D
import torch.nn as nn
from transformers import AutoConfig
import torch.autograd

@dataclass
class QuantBlockConfig():
    Attention_W_bit: int = 32
    Attention_A_bit: int = 32
    Attention_KV_bit: int = 32
    Attention_A_layerwise: bool = False
    Attention_W_layerwise: bool = True
    Attention_KV_layerwise: bool = False
    Attention_layerwise: bool = False

    MLP_W_bit: int = 32
    MLP_A_bit: int = 32
    MLP_A_layerwise: bool = False
    MLP_W_layerwise: bool = True
    MLP_layerwise: bool = False

    gradclip: tuple = (-2, 2)

    @classmethod
    def from_dict(cls, d: dict) -> "QuantBlockConfig":
        """Create a QuantBlockConfig from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

class SymQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_bits, layerwise: bool = False, clip_val: tuple = None):
        ctx.save_for_backward(input)
        ctx.clip_val = clip_val 
        
        if layerwise:
            max_val = torch.max(torch.abs(input))
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_val = (
                    torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )
            # (batch, seq_len, num_heads, head_dim)
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.reshape(input.shape[0], input.shape[1], -1)
                max_val = (
                    torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError

        # quantize and dequantize the input
        # we add a small epsilon to avoid division by zero
        alpha = max_val / ((2**(num_bits - 1) - 1) + 1e-6)
        X_q = torch.round(input / alpha) * alpha

        return X_q.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        clip_val = ctx.clip_val

        # clips the output (effectively STE)
        grad_input = grad_output.clone()
        if clip_val is not None:
            grad_input[input > clip_val[1]] = 0
            grad_input[input < clip_val[0]] = 0
        return grad_input, None, None, None


class QuantLinear(nn.Module):
    def __init__(self, layer, W_bit: int = 32, A_bit: int = 32, W_layerwise: bool = True, A_layerwise: bool = True, gradclip: tuple = None):
        super().__init__()
        self.in_features = layer.weight.shape[0]
        self.out_features = layer.weight.shape[1]

        self.W_bit = W_bit
        self.A_bit = A_bit
        self.W_layerwise = W_layerwise
        self.A_layerwise = A_layerwise
        self.gradclip = gradclip
        
        self.weight = nn.Parameter(layer.weight.data.clone())
        self.bias = nn.Parameter(layer.bias.data.clone()) if layer.bias is not None else None
        self.quantFunc = SymQuantization.apply
        self.is_conv1d = isinstance(layer, Conv1D)

    def forward(self, x):

        weight = self.weight
        if self.W_bit and self.W_bit < 32:
            weight = self.quantFunc(self.weight, self.W_bit, 
                                       self.W_layerwise, self.gradclip)

        act = x
        if self.A_bit and self.A_bit < 32:
            act = self.quantFunc(x, self.A_bit, 
                                    self.A_layerwise, self.gradclip)


        out = act @ weight
        if self.bias is not None:
            out = out + self.bias
        return out.contiguous()

def quantize_model(model, quant_configs):
    # Import here to avoid circular import
    from _transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2MLPQ, GPT2AttentionQ
    
    for name, module in list(model.named_modules()):
        class_name = type(module).__name__

        parent = model
        name_parts = name.split('.')
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        attr_name = name_parts[-1]
        
        # ['transformer', 'h', 'block_idx', ...]
        if len(name_parts) >= 3 and name_parts[0] == 'transformer' and name_parts[1] == 'h':
            block_idx = int(name_parts[2])
            quant_config = quant_configs[block_idx]
        else:
            continue 
        
        if class_name == "GPT2MLP":
            new_module = GPT2MLPQ(
                intermediate_size=module.c_fc.weight.shape[1],
                config=model.config,
                quant_config=quant_config
            )

            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(parent, attr_name, new_module)
        
        elif class_name == "GPT2Attention":
            new_module = GPT2AttentionQ(
                config=model.config,
                is_cross_attention=module.is_cross_attention,
                layer_idx=module.layer_idx,
                quant_config=quant_config
            )
            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(parent, attr_name, new_module)

def set_active_quant_config(singleton_config:dict[int, QuantBlockConfig], target_config:dict[int, QuantBlockConfig]):
    for idx, tgt_cfg in target_config.items():
        if idx in singleton_config:
            sing_cfg = singleton_config[idx]
            sing_cfg.Attention_W_bit = tgt_cfg.Attention_W_bit
            sing_cfg.Attention_A_bit = tgt_cfg.Attention_A_bit
            sing_cfg.Attention_KV_bit = tgt_cfg.Attention_KV_bit
            sing_cfg.Attention_A_layerwise = tgt_cfg.Attention_A_layerwise
            sing_cfg.Attention_W_layerwise = tgt_cfg.Attention_W_layerwise
            sing_cfg.Attention_KV_layerwise = tgt_cfg.Attention_KV_layerwise
            sing_cfg.Attention_layerwise = tgt_cfg.Attention_layerwise
            sing_cfg.MLP_W_bit = tgt_cfg.MLP_W_bit
            sing_cfg.MLP_A_bit = tgt_cfg.MLP_A_bit
            sing_cfg.MLP_A_layerwise = tgt_cfg.MLP_A_layerwise
            sing_cfg.MLP_W_layerwise = tgt_cfg.MLP_W_layerwise
            sing_cfg.MLP_layerwise = tgt_cfg.MLP_layerwise
            sing_cfg.gradclip = tgt_cfg.gradclip
import torch
from hbdk4.compiler import leap

from leap_llm.nn.modules import FakeQuantLinear
from leap_llm.nn.utils import Module


class GemmaMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = FakeQuantLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = FakeQuantLinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = FakeQuantLinear(self.intermediate_size, self.hidden_size, bias=False)
        
    def build(self, hidden_state):
        hidden_state = leap.cast_type(hidden_state, output_type=leap.float32)
        x = self.gate_proj(hidden_state)
        x = leap.cast_type(x, output_type=leap.float16)
        x = leap.gelu(x)
        x = leap.cast_type(x, output_type=leap.float32)
        up_proj_h = self.up_proj(hidden_state)
        up_proj_h = leap.cast_type(up_proj_h, output_type=leap.float16)
        x = leap.mul(x, up_proj_h)
        x = leap.cast_type(x, output_type=leap.float32)
        x = self.down_proj(x)
        x = leap.cast_type(x, output_type=leap.float16)
        return x
        
    def forward(self, hidden_state):
        x = self.gate_proj(hidden_state)
        x = torch.nn.functional.gelu(x)
        up_proj_h = self.up_proj(hidden_state)
        x = torch.mul(x, up_proj_h)
        return self.down_proj(x)

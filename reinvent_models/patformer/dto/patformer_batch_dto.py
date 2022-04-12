from dataclasses import dataclass
import torch


@dataclass
class PatformerBatchDTO:
    input: torch.Tensor
    input_mask: torch.Tensor
    output: torch.Tensor
    output_mask: torch.Tensor
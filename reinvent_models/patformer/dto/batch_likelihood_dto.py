from dataclasses import dataclass
from typing import Union
import torch

from reinvent_models.link_invent.dto.linkinvent_batch_dto import LinkInventBatchDTO
from reinvent_models.patformer.dto.patformer_batch_dto import PatformerBatchDTO


@dataclass
class BatchLikelihoodDTO:
    batch: Union[PatformerBatchDTO, LinkInventBatchDTO]
    likelihood: torch.Tensor
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn
from transformers import AutoModel
from transformers.file_utils import ModelOutput

from .base_model import BaseEncoderOutput, BaseModel

logger = logging.getLogger(__name__)


@dataclass
class InfoBatchEncoderOutput(BaseEncoderOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    query_indices: Optional[Tensor] = None


class InfoBatchModel(BaseModel):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = None,
        normalized: bool = False,
        sentence_pooling_method: str = "cls",
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        use_inbatch_neg: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            normalized=normalized,
            sentence_pooling_method=sentence_pooling_method,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            use_inbatch_neg=use_inbatch_neg,
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        query_indices: Dict[str, Tensor] = None,
        teacher_score: Tensor = None,
    ):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = (
                    self.compute_similarity(
                        q_reps[
                            :,
                            None,
                            :,
                        ],
                        p_reps.view(q_reps.size(0), group_size, -1),
                    ).squeeze(1)
                    / self.temperature
                )  # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None

        query_indices = self._dist_gather_tensor(query_indices)

        return InfoBatchEncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            query_indices=query_indices,
        )

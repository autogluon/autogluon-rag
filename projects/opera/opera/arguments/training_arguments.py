from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class OperaTrainingArguments(TrainingArguments):
    output_dir: str = field(default=None, metadata={"help": "overtwrite"})
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(
        default=False, metadata={"help": "Freeze the parameters of position embeddings"}
    )
    sentence_pooling_method: str = field(default="cls", metadata={"help": "the pooling method, should be cls or mean"})
    normalized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "enable gradient checkpointing for memory efficiency"}
    )

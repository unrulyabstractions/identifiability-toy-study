from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class InductionModelConfig(BaseModel):
    vocab_size: PositiveInt
    seq_len: PositiveInt
    d_model: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    ff_fanout: PositiveInt
    use_ff: bool
    use_pos_encoding: bool
    use_layer_norm: bool
    device: str = "cpu"


class InductionHeadsTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    ih_model_config: InductionModelConfig
    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    lr_warmup: int | float
    weight_decay: float
    lr_schedule: Literal["cosine", "constant", "linear"] = "linear"
    seed: int = 0
    attention_maps_n_steps: PositiveInt
    prefix_window: PositiveInt


class IHTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["induction_head"]
    prefix_window: PositiveInt | None = Field(
        default=None,
        description="Number of tokens to use as a prefix window for the induction head. If none, uses the full sequence length.",
    )

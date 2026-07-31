from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class ArchitectureConfig(BaseModel):
    """Configuration for the production CoordinateLSTM."""
    text_embedding_dim: int = Field(default=128, ge=16)
    text_encoder_hidden_dim: int = Field(default=256, ge=16)
    text_encoder_layers: int = Field(default=2, ge=1)
    
    decoder_input_dim: int = Field(default=3, ge=1)
    decoder_hidden_dim: int = Field(default=512, ge=16)
    decoder_layers: int = Field(default=3, ge=1)
    
    mdn_mixtures: int = Field(default=20, ge=1)
    
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    layer_norm: bool = Field(default=True)
    residual_connections: bool = Field(default=True)
    
    # If a style vector is eventually passed, what size is it?
    style_dim: int = Field(default=0, ge=0)

class TrainingConfig(BaseModel):
    """Configuration for the training loop."""
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=100, ge=1)
    
    # Optimizer & Scheduler
    optimizer: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    weight_decay: float = Field(default=1e-4, ge=0.0)
    scheduler_patience: int = Field(default=5, ge=1)
    scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    
    # Stability
    gradient_clip_val: float = Field(default=5.0, gt=0.0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    mixed_precision: bool = Field(default=True)
    
    # Scheduled Sampling (Teacher Forcing)
    teacher_forcing_start: float = Field(default=1.0, ge=0.0, le=1.0)
    teacher_forcing_end: float = Field(default=0.5, ge=0.0, le=1.0)
    teacher_forcing_decay_epochs: int = Field(default=50, ge=1)
    
    # Data length
    max_sequence_length: int = Field(default=2000, ge=1)

class ProductionConfig(BaseModel):
    """Root configuration object."""
    architecture: ArchitectureConfig = ArchitectureConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Metadata
    seed: int = Field(default=42)
    checkpoint_dir: str = Field(default="checkpoints/")
    log_dir: str = Field(default="logs/")

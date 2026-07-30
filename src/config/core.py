import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class DatasetConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    datasets_dir: str = "data/datasets"
    max_sequence_length: int = 2000

class ModelConfig(BaseModel):
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

class TrainingConfig(BaseModel):
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 100
    mixed_precision: bool = True
    output_dir: str = "outputs"

class ProjectConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

def load_config(yaml_path: str | Path) -> ProjectConfig:
    """Loads configuration from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return ProjectConfig(**(data or {}))

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.datasets.structures import TrajectorySample
from src.inference.config import InferenceConfig

class InferenceResult(BaseModel):
    """
    Canonical output object returned by the InferencePipeline.
    Encapsulates all generated data, metadata, warnings, and latency logs.
    """
    input_text: str = Field(description="Original requested string.")
    normalized_text: str = Field(description="Preprocessed string passed to tokenizer.")
    trajectory: TrajectorySample = Field(description="The finalized, post-processed canonical geometry.")
    
    # Execution Metadata
    configuration: InferenceConfig = Field(description="The specific config used during generation.")
    timing: Dict[str, float] = Field(default_factory=dict, description="Latency in ms per pipeline stage.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional framework telemetry.")
    cache_statistics: Dict[str, bool] = Field(default_factory=dict, description="Hit/Miss logs for memoization.")
    
    # Outcomes
    export_paths: Dict[str, str] = Field(default_factory=dict, description="Mapping of format to absolute file path.")
    warnings: List[str] = Field(default_factory=list, description="Recoverable warnings encountered during execution.")
    errors: List[str] = Field(default_factory=list, description="Hard failures preventing export.")

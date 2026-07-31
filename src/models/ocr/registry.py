from typing import Type, Dict, Any
from src.models.ocr.base import BaseOCRModel

_OCR_MODELS: Dict[str, Type[BaseOCRModel]] = {}

def register_ocr_model(name: str):
    """
    Decorator to register an OCR model class in the registry.
    """
    def wrapper(cls: Type[BaseOCRModel]):
        _OCR_MODELS[name] = cls
        return cls
    return wrapper

def build_ocr_model(name: str, vocab_size: int, config: Dict[str, Any]) -> BaseOCRModel:
    """
    Instantiates an OCR model by its registered name.
    """
    if name not in _OCR_MODELS:
        raise ValueError(f"OCR model '{name}' not found. Available: {list(_OCR_MODELS.keys())}")
    return _OCR_MODELS[name](vocab_size, config)

# Import models to ensure registration
import src.models.ocr.crnn

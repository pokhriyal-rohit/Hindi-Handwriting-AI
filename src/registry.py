from typing import Any, Callable, Dict, Type

class Registry:
    """
    Plugin architecture registry.
    Allows dynamic registration of models, datasets, renderers, representations, and rendering plugins.
    """
    _registry: Dict[str, Dict[str, Type[Any]]] = {
        "models": {},
        "datasets": {},
        "renderers": {},
        "representations": {},
        "scalers": {},
        "layouts": {},
        "smoothers": {},
        "pressure_models": {},
        "ink_models": {},
        "exporters": {}
    }

    @classmethod
    def get(cls, category: str, name: str) -> Type[Any]:
        if category not in cls._registry:
            raise ValueError(f"Category '{category}' not found in registry.")
        if name not in cls._registry[category]:
            return None
        return cls._registry[category][name]

    # --- Core AI Plugins ---
    @classmethod
    def register_scaler(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["scalers"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def register_model(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["models"][name] = wrapped_class
            return wrapped_class
        return wrapper
        
    @classmethod
    def register_dataset(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["datasets"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def register_representation(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["representations"][name] = wrapped_class
            return wrapped_class
        return wrapper

    # --- Rendering Engine Plugins ---
    @classmethod
    def register_layout(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["layouts"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_layout(cls, name: str) -> Type[Any]:
        return cls.get("layouts", name)

    @classmethod
    def register_smoother(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["smoothers"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_smoother(cls, name: str) -> Type[Any]:
        return cls.get("smoothers", name)

    @classmethod
    def register_pressure_model(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["pressure_models"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_pressure_model(cls, name: str) -> Type[Any]:
        return cls.get("pressure_models", name)

    @classmethod
    def register_ink_model(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["ink_models"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_ink_model(cls, name: str) -> Type[Any]:
        return cls.get("ink_models", name)

    @classmethod
    def register_exporter(cls, name: str) -> Callable:
        def wrapper(wrapped_class: Type[Any]) -> Type[Any]:
            cls._registry["exporters"][name] = wrapped_class
            return wrapped_class
        return wrapper

    @classmethod
    def get_exporter(cls, name: str) -> Type[Any]:
        return cls.get("exporters", name)

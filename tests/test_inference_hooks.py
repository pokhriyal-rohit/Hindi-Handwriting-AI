from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline
from src.inference.hooks import BaseHook, HookContext
from src.registry import Registry

@Registry.register_hook("test_hook")
class TestHook(BaseHook):
    def __init__(self):
        self.calls = {}
        
    def before_inference(self, ctx: HookContext): self.calls["before_inference"] = True
    def after_prediction(self, ctx: HookContext): self.calls["after_prediction"] = True
    def before_rendering(self, ctx: HookContext): self.calls["before_rendering"] = True
    def after_inference(self, ctx: HookContext): self.calls["after_inference"] = True

def test_inference_hooks():
    config = InferenceConfig(model_name="dummy_predictor", hooks=["test_hook"])
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    # Generate
    result = pipeline.generate("HOOK_TEST")
    
    # Grab the hook instance from the session
    hook_instance = session.hooks[0]
    
    assert hook_instance.calls.get("before_inference") is True
    assert hook_instance.calls.get("after_prediction") is True
    assert hook_instance.calls.get("before_rendering") is True
    assert hook_instance.calls.get("after_inference") is True
    
    session.shutdown()

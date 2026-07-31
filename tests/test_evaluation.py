from src.registry import Registry
from src.evaluation.metrics.base import BaseMetric
from tests.test_renderer import dummy_trajectory

@Registry.register_metric("dummy_metric")
class DummyMetric(BaseMetric):
    @classmethod
    def name(cls) -> str:
        return "Dummy"
        
    @classmethod
    def version(cls) -> str:
        return "1.0.0"
        
    @classmethod
    def description(cls) -> str:
        return "A dummy metric for testing the registry."
        
    def evaluate(self, prediction, target):
        self.validate(prediction, target)
        return {"score": 1.0}
        
    def summarize(self, results):
        return {"mean_score": 1.0}

def test_metric_registry():
    metric_cls = Registry.get_metric("dummy_metric")
    assert metric_cls is not None
    assert metric_cls.name() == "Dummy"
    
def test_base_metric_validation():
    metric = DummyMetric()
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    # Should pass
    assert metric.validate(traj1, traj2)
    
    # Should fail
    try:
        metric.validate(traj1, "invalid_target")
        assert False, "Should raise TypeError"
    except TypeError:
        pass

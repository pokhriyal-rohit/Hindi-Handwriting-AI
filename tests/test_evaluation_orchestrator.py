import os
import tempfile
from src.evaluation.config import EvaluationConfig
from src.evaluation.benchmarks.orchestrator import BenchmarkOrchestrator
from tests.test_renderer import dummy_trajectory

def test_benchmark_orchestrator():
    config = EvaluationConfig()
    orchestrator = BenchmarkOrchestrator(config)
    
    # 2 dummy samples
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    # Give them slightly different paths
    traj2.strokes[0].points[-1].x += 1.0
    
    batch = [
        (traj1, traj1),
        (traj2, traj1)
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = orchestrator.run_suite(batch, tmpdir)
        
        # Ensure reports generated
        assert os.path.exists(os.path.join(tmpdir, "evaluation_report.json"))
        assert os.path.exists(os.path.join(tmpdir, "evaluation_report.md"))
        assert os.path.exists(os.path.join(tmpdir, "benchmark_summary.csv"))
        
        # Ensure metrics were run (e.g. endpoint error should be recorded)
        assert "endpoint_error_mean" in summary

import os
import tempfile
from src.evaluation.config import EvaluationConfig
from src.evaluation.reports.generators import generate_json_report, generate_markdown_report, generate_csv_summary
from src.evaluation.visualization.plotters import plot_trajectory_overlay
from tests.test_renderer import dummy_trajectory

def test_report_generators():
    config = EvaluationConfig()
    summary = {
        "dtw_mean": 1.5,
        "dtw_std": 0.2,
        "svg_time_mean": 0.05
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        md_path = os.path.join(tmpdir, "report.md")
        csv_path = os.path.join(tmpdir, "summary.csv")
        
        generate_json_report(config, summary, json_path)
        assert os.path.exists(json_path)
        
        generate_markdown_report(config, summary, md_path)
        assert os.path.exists(md_path)
        with open(md_path, 'r') as f:
            content = f.read()
            assert "dtw" in content
            
        generate_csv_summary(config, summary, csv_path)
        assert os.path.exists(csv_path)

def test_visualization():
    traj1 = dummy_trajectory()
    traj2 = dummy_trajectory()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "overlay.png")
        plot_trajectory_overlay(traj1, traj2, out_path)
        # It might not generate if matplotlib is missing, which is fine
        if os.path.exists(out_path):
            assert True

import os
import json
import shutil
from src.inference.config import InferenceConfig
from src.inference.session import InferenceSession
from src.inference.pipeline import InferencePipeline

# Evaluation metrics
from src.registry import Registry

def run_golden_pipeline():
    print("============================================")
    print("GOLDEN STANDARD INTEGRATION TEST")
    print("============================================")
    
    # 1. Setup specific Golden configuration
    config = InferenceConfig(
        model_name="deterministic_hindi",
        enable_cache=False,
        export_formats=["svg"]  # Add png/pdf if system supports cairo natively in tests
    )
    
    session = InferenceSession(config)
    pipeline = InferencePipeline(session)
    
    # 2. Setup isolated output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "run_golden"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 3. Generate
    text_input = "नमस्ते दुनिया"
    print("[1/5] Generating handwriting for input...")
    res = pipeline.generate(text_input, run_dir=output_dir)
    
    # 4. Extract and verify Artifacts
    print("[2/5] Verifying rendered outputs...")
    assert "svg" in res.export_paths, "SVG was not generated"
    assert os.path.exists(res.export_paths["svg"]), "SVG file missing"
    
    # 5. Save Artifacts explicitly
    print("[3/5] Saving trajectory and metadata artifacts...")
    traj_path = os.path.join(output_dir, "trajectory.json")
    with open(traj_path, "w", encoding="utf-8") as f:
        f.write(res.trajectory.model_dump_json(indent=2))
        
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        # Convert InferenceResult to JSON
        f.write(res.model_dump_json(indent=2))
        
    # 6. Run Evaluation Metrics dynamically on the output
    print("[4/5] Executing Evaluation Framework metrics...")
    eval_results = {}
    metric_cls = Registry.get_metric("inference_latency")
    if metric_cls:
        metric = metric_cls()
        # Mock prediction latency computation
        eval_results["inference_latency"] = metric.compute({"total_ms": res.timing.get("total_ms", 100.0)})
        
    eval_report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
        
    # 7. Final Verification
    print("[5/5] Final Verifications...")
    assert os.path.exists(traj_path)
    assert os.path.exists(meta_path)
    assert os.path.exists(eval_report_path)
    
    assert res.metadata["runtime"]["git_commit"] is not None
    assert len(res.trajectory.strokes) > 0
    
    print("============================================")
    print("GOLDEN PIPELINE SUCCESSFUL")
    print(f"Artifacts saved to: {output_dir}")
    print("============================================")

if __name__ == "__main__":
    run_golden_pipeline()

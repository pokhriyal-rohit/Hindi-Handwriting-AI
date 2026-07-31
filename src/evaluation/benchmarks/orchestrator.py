from typing import List, Tuple, Dict, Any
from src.registry import Registry
from src.datasets.structures import TrajectorySample
from src.evaluation.config import EvaluationConfig
from src.evaluation.reports.generators import generate_json_report, generate_markdown_report, generate_csv_summary

class BenchmarkOrchestrator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = []
        
        # Instantiate all registered metrics
        for name, cls in Registry._registry["metrics"].items():
            metric = cls()
            self.metrics.append(metric)
            # Track version in config
            self.config.metric_versions[name] = metric.version()
            
    def run_suite(self, samples: List[Tuple[TrajectorySample, TrajectorySample]], output_dir: str):
        """
        Executes the full benchmark suite over a batch of <Prediction, Target> pairs.
        """
        all_results: Dict[str, List[Dict[str, Any]]] = {m.name(): [] for m in self.metrics}
        
        # 1. Batch Execution
        for pred, tgt in samples:
            for metric in self.metrics:
                try:
                    res = metric.evaluate(pred, tgt)
                    all_results[metric.name()].append(res)
                except Exception as e:
                    # Log failure but continue batch
                    print(f"Metric {metric.name()} failed on sample {pred.sample_id}: {e}")
                    
        # 2. Statistical Aggregation
        summary = {}
        for metric in self.metrics:
            try:
                metric_summary = metric.summarize(all_results[metric.name()])
                summary.update(metric_summary)
            except Exception as e:
                print(f"Metric {metric.name()} failed summarization: {e}")
                
        # 3. Report Generation
        import os
        os.makedirs(output_dir, exist_ok=True)
        generate_json_report(self.config, summary, os.path.join(output_dir, "evaluation_report.json"))
        generate_markdown_report(self.config, summary, os.path.join(output_dir, "evaluation_report.md"))
        generate_csv_summary(self.config, summary, os.path.join(output_dir, "benchmark_summary.csv"))
        
        return summary

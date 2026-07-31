import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from .parser import IIITHWParser

class DatasetAnalyzer:
    """Analyzes a dataset directory and generates a statistical report and plots."""
    
    def __init__(self, dataset_path: str | Path, output_dir: str | Path):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            "total_files": 0,
            "image_files": 0,
            "trajectory_files": 0,
            "total_strokes": 0,
            "total_points": 0,
            "corrupted_files": 0
        }
        self.stroke_counts = []
        self.point_counts = []
        self.delta_x = []
        self.delta_y = []
        
    def run_analysis(self):
        """Iterates through the dataset and collects statistics."""
        print(f"Starting analysis on {self.dataset_path}...")
        
        if not self.dataset_path.exists():
            print(f"Dataset path does not exist: {self.dataset_path}")
            return
            
        all_files = list(self.dataset_path.rglob("*.*"))
        self.stats["total_files"] = len(all_files)
        
        for file in all_files:
            ext = file.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                self.stats["image_files"] += 1
            elif ext in ['.txt', '.xml']:
                self.stats["trajectory_files"] += 1
                self._analyze_trajectory(file)
            else:
                pass # Other files
                
        self._generate_report()
        self._generate_plots()
        
    def _analyze_trajectory(self, file: Path):
        """Analyzes a single trajectory file."""
        try:
            # Attempt to parse using our skeleton parser
            traj = IIITHWParser.parse_txt_file(file)
            if not traj.strokes:
                return
                
            self.stats["total_strokes"] += len(traj.strokes)
            self.stroke_counts.append(len(traj.strokes))
            
            pts = 0
            for stroke in traj.strokes:
                pts += len(stroke.points)
                
                # Calculate deltas for this stroke
                for i in range(1, len(stroke.points)):
                    self.delta_x.append(stroke.points[i].x - stroke.points[i-1].x)
                    self.delta_y.append(stroke.points[i].y - stroke.points[i-1].y)
                    
            self.stats["total_points"] += pts
            self.point_counts.append(pts)
            
        except Exception as e:
            self.stats["corrupted_files"] += 1
            
    def _generate_report(self):
        """Generates a markdown report."""
        report_path = self.output_dir / "dataset_analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Dataset Analysis Report\n\n")
            f.write(f"**Dataset Location:** `{self.dataset_path}`\n\n")
            f.write("## File Statistics\n")
            f.write(f"- **Total Files:** {self.stats['total_files']}\n")
            f.write(f"- **Image Files (Offline):** {self.stats['image_files']}\n")
            f.write(f"- **Trajectory Files (Online):** {self.stats['trajectory_files']}\n")
            f.write(f"- **Corrupted/Unparseable Files:** {self.stats['corrupted_files']}\n\n")
            
            f.write("## Trajectory Statistics\n")
            if self.stroke_counts:
                f.write(f"- **Total Strokes Analyzed:** {self.stats['total_strokes']}\n")
                f.write(f"- **Average Strokes per Sample:** {np.mean(self.stroke_counts):.2f}\n")
                f.write(f"- **Average Points per Stroke:** {self.stats['total_points'] / self.stats['total_strokes']:.2f}\n")
                f.write(f"- **Max Strokes in a Sample:** {np.max(self.stroke_counts)}\n")
            else:
                f.write("- *No trajectory files found. The dataset appears to be entirely offline (images).* \n")
                f.write("- **CRITICAL WARNING:** Trajectory synthesis (SVG, pen state, animated drawing) requires online coordinate data. This dataset cannot be used for the final generative model.\n")
                
        print(f"Report generated at {report_path}")

    def _generate_plots(self):
        """Generates statistical plots."""
        if not self.stroke_counts:
            return # Nothing to plot
            
        plt.figure(figsize=(10, 5))
        plt.hist(self.stroke_counts, bins=20, color='skyblue', edgecolor='black')
        plt.title('Distribution of Strokes per Sample')
        plt.xlabel('Number of Strokes')
        plt.ylabel('Frequency')
        plt.savefig(self.output_dir / 'stroke_distribution.png')
        plt.close()
        
        if self.delta_x and self.delta_y:
            plt.figure(figsize=(10, 5))
            plt.scatter(self.delta_x[:1000], self.delta_y[:1000], alpha=0.5, s=1)
            plt.title('Delta Movement Distribution (Sample)')
            plt.xlabel('Delta X')
            plt.ylabel('Delta Y')
            plt.savefig(self.output_dir / 'delta_distribution.png')
            plt.close()

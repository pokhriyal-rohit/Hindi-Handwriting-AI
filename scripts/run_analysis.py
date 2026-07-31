import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.analysis import DatasetAnalyzer

def main():
    data_dir = Path("data/raw/IIIT-HW-Hindi_v1")
    output_dir = Path("outputs/analysis")
    
    analyzer = DatasetAnalyzer(data_dir, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.synthetic.generator.synthetic_trajectory_generator import SyntheticTrajectoryGenerator

def main():
    try:
        ds = SyntheticTrajectoryGenerator()
        ds.load()
        print("Validation:", ds.validate())
        print("Analysis:", ds.analyze())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

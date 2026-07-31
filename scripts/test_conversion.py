import os
from src.datasets.converters import CustomCollectorConverter
from src.renderer.pipeline import RenderingEngine
from src.renderer.config import RenderingConfig

def test_conversion():
    sample_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "custom_hindi", "writer_mock", "क", "sample_001.json"))
    out_file = os.path.join(os.path.dirname(sample_file), "render_test.svg")
    
    # 1. Test Conversion
    try:
        sample = CustomCollectorConverter.from_json(sample_file)
        print("Successfully converted raw JSON to TrajectorySample.")
        print(f"Loaded {len(sample.strokes)} strokes.")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return
        
    # 2. Test Rendering
    try:
        renderer = RenderingEngine(RenderingConfig())
        renderer.render(sample, out_file, format="svg")
        print(f"Successfully rendered to {out_file}")
    except Exception as e:
        print(f"Rendering failed: {e}")
        
if __name__ == "__main__":
    test_conversion()

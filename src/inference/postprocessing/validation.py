import copy
from src.registry import Registry
from src.inference.postprocessing.base import BasePostProcessor
from src.datasets.structures import TrajectorySample

@Registry.register_postprocessor("coordinate_clamp")
class CoordinateClampProcessor(BasePostProcessor):
    """
    Ensures that coordinates are within safe bounds to prevent SVG explosion bugs.
    """
    def process(self, sample: TrajectorySample) -> TrajectorySample:
        # Create explicit copy as mandated by contract
        new_sample = copy.deepcopy(sample)
        
        for stroke in new_sample.strokes:
            for pt in stroke.points:
                # Clamp coordinates to a huge but safe bounding box to prevent rendering issues
                pt.x = max(min(pt.x, 10000.0), -10000.0)
                pt.y = max(min(pt.y, 10000.0), -10000.0)
                
        return new_sample

@Registry.register_postprocessor("metadata_enricher")
class MetadataEnricherProcessor(BasePostProcessor):
    """
    Appends execution pipeline metadata directly to the payload extensions.
    """
    def process(self, sample: TrajectorySample) -> TrajectorySample:
        new_sample = copy.deepcopy(sample)
        new_sample.extensions["post_processed"] = True
        return new_sample

"""Dataset adapters that emit canonical samples."""
from compression_pipeline.adapters.turb_rot_npz import TurbRotNPZAdapter
from compression_pipeline.adapters.e3sm_npz import E3SMNPZAdapter

__all__ = ["TurbRotNPZAdapter"]

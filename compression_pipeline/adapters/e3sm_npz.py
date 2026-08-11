from __future__ import annotations

from pathlib import Path

from compression_pipeline.adapters.turb_rot_npz import TurbRotNPZAdapter


class E3SMNPZAdapter(TurbRotNPZAdapter):
    """Reads preprocessed E3SM NPZ data in [V,S,T,H,W] layout."""

    def __init__(
        self,
        data_root: str | Path,
        dataset_id: str = "e3sm_npz",
        section_index: int = 0,
        section_start: int = 0,
        time_start: int = 0,
        image_group_mode: str = "auto",
        image_channel_count: int | None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_id=dataset_id,
            section_index=section_index,
            section_start=section_start,
            time_start=time_start,
            image_group_mode=image_group_mode,
            image_channel_count=image_channel_count,
        )

    def _timestamp(self, index: int) -> str:
        return f"e3sm_t{index:04d}"

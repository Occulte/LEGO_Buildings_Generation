from typing import Dict, List
from Utils.ModelUtils.Connection.connpoint import CPoint
import os
import numpy as np
import json

from Utils.ModelUtils.unit import Unit

# TODO: Change the directories of these static files.
_voxel_info = {}
_voxel_info_path = os.path.join("Static", "brick_voxel.json")

with open(_voxel_info_path, "r") as f:
    _voxel_info = json.load(f)


_cpoint_voxel_info = {}
_cpoint_voxel_info_path = os.path.join("Static", "cpoint_voxel.json")
with open(_cpoint_voxel_info_path, "r") as f:
    _cpoint_voxel_info = json.load(f)


class BrickTemplate:
    def __init__(
        self,
        c_points: List[CPoint],
        bboxes: List[Dict],
        id: str,
    ) -> None:
        self.c_points = c_points
        self.bboxes = bboxes
        self.id = id

        self.units: List[Unit] = None
        self.whole_bbox: Dict = None

        self.set_voxel_information()

    def set_voxel_information(self):
        # 1. set occupancy voxels
        brick_id = self.id
        if (
            brick_id not in _voxel_info.keys()
            or brick_id not in _cpoint_voxel_info.keys()
            or "stud_voxels" not in _cpoint_voxel_info[brick_id].keys()
            or "tube_voxels" not in _cpoint_voxel_info[brick_id].keys()
        ):
            return

        self.voxels = np.array(_voxel_info[brick_id], dtype=np.int32).reshape((-1, 3))
        self.stud_voxels = np.array(
            _cpoint_voxel_info[brick_id]["stud_voxels"], dtype=np.int32
        ).reshape((-1, 3))
        self.tube_voxels = np.array(
            _cpoint_voxel_info[brick_id]["tube_voxels"], dtype=np.int32
        ).reshape((-1, 3))
        self.stud_voxel_augmented = np.array(
            _cpoint_voxel_info[brick_id]["stud_voxel_augmented"], dtype=np.int32
        )
        self.tube_voxel_augmented = np.array(
            _cpoint_voxel_info[brick_id]["tube_voxel_augmented"], dtype=np.int32
        )

"""
This should be a temp file
"""

import os
import zipfile

import numpy as np
from typing import List

from Utils.GeometryUtils.geometry_utils import add_vertices_and_triangles_from_voxel
from Utils.ModelUtils.brick_instance import BrickInstance
from Utils.ModelUtils.Connection.connpoint import CPoint

from Utils.BrickSet.brick_ids import (
    brick_id_to_class_id,
    brick_id_to_non_reduced_class_id,
    # brick_id_to_non_reduced_for_visualization,
    does_brick_need_augmented_conn_voxel,
)


# TODO: Need to complete the read-write functions of the bricks.

# Need to implement a function to read from an ldr file
# from bricks_modeling.bricks.model import Model
# from bricks_modeling.file_IO.model_reader import read_model_from_file
from Utils.ModelUtils.IO.io_util import write_bricks_to_file

import matplotlib.pyplot as plt
import open3d as o3d

from Utils.GenUtils.color_utils import color_palette_paper


def get_brick_occupancy(brick: BrickInstance, voxel_offsets: np.ndarray) -> np.ndarray:
    brick_voxels = brick.get_brick_voxels()

    occupied_voxels: np.ndarray = brick_voxels + voxel_offsets
    assert np.all(np.round(occupied_voxels) % 4 == 0)
    occupied_voxels = occupied_voxels // 4
    occupied_voxels[:, 1] = -occupied_voxels[:, 1]
    return occupied_voxels


def get_brick_voxel_components(brick: BrickInstance, voxel_offsets: np.ndarray):
    brick_voxels = get_brick_occupancy(brick, voxel_offsets)
    stud_voxels, tube_voxels = get_brick_conn_occupancy(brick, voxel_offsets)

    brick_voxels = np.array(
        [
            v
            for v in brick_voxels.tolist()
            if v not in stud_voxels.tolist() and v not in tube_voxels.tolist()
        ]
    )

    return brick_voxels, stud_voxels, tube_voxels


def get_brick_conn_occupancy(brick: BrickInstance, voxel_offsets: np.ndarray):
    brick_stud_voxels, brick_tube_voxels = brick.get_stud_tube_voxels()

    stud_voxels = brick_stud_voxels + voxel_offsets
    assert np.all(np.round(stud_voxels) % 4 == 0)
    stud_voxels = stud_voxels // 4
    stud_voxels[:, 1] = -stud_voxels[:, 1]
    tube_voxels = brick_tube_voxels + voxel_offsets
    assert np.all(np.round(tube_voxels) % 4 == 0)
    tube_voxels = tube_voxels // 4
    tube_voxels[:, 1] = -tube_voxels[:, 1]
    return stud_voxels, tube_voxels


def add_voxels_to_occupancy_map(
    brick: BrickInstance,
    occupancy_map: np.ndarray,
    voxel_offset=np.ndarray([0, 0, 0], dtype=np.int32),
    simplex_reduction=True,
):
    """
    brick: A brick instance
    occupancy_map: the voxel volume for assigning the occupancy
    voxel_offset: the offset of the brick in the voxel volume
    simplex_reduction: whether to reduce to atomic bricks
    used_for_paper_rendering: whether to use the class_ids defined in brick_id_to_non_reduced_for_visualization?
    """
    occupied_voxels = get_brick_occupancy(brick, voxel_offset)
    if occupied_voxels is None:
        return False

    # check if the voxels are within the occupancy map
    assert np.min(occupied_voxels) >= 0
    assert np.all(np.max(occupied_voxels, axis=0) - np.array(occupancy_map.shape) <= 0)

    if simplex_reduction:
        occupancy_map[
            occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]
        ] = brick_id_to_class_id[brick.template.id]
    else:
        occupancy_map[
            occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]
        ] = brick_id_to_non_reduced_class_id[brick.template.id]
    return True


def get_conn_voxels_mask(
    bricks: List[BrickInstance], mask_shape, voxel_offsets, reduce_simplex=True
):
    """
    In this function, we define the conn voxels as the following
    1. For each conn point, its surrounding voxels within a distance of 7 to the conn point;
    2. For special bricks, we add strip voxels in-between the conn points for differentiating brick orientations.
    augment: a boolean controlling whether to add strip voxels stated in the second point.
    3. Note that reduce_simplex means that we need to disable augmented connection voxels for 3022, 3023, 3069b, 3068b, etc.
    """
    mask = np.zeros(mask_shape, dtype=np.uint8)

    stud_voxels = np.zeros((0, 3), dtype=np.int32)
    tube_voxels = np.zeros((0, 3), dtype=np.int32)

    if reduce_simplex:
        does_brick_need_augmented_conn_voxel["3022"] = False
        does_brick_need_augmented_conn_voxel["3023"] = False
        does_brick_need_augmented_conn_voxel["3623"] = False
        does_brick_need_augmented_conn_voxel["3069b"] = False
        does_brick_need_augmented_conn_voxel["3068b"] = False
    else:
        does_brick_need_augmented_conn_voxel["3022"] = True
        does_brick_need_augmented_conn_voxel["3023"] = True
        does_brick_need_augmented_conn_voxel["3623"] = True
        does_brick_need_augmented_conn_voxel["3069b"] = True
        does_brick_need_augmented_conn_voxel["3068b"] = True

    for brick in bricks:
        brick_stud_voxels, brick_tube_voxels = brick.get_stud_tube_voxels()

        # transform the voxels to the global coordinates

        stud_voxels = np.concatenate([stud_voxels, brick_stud_voxels], axis=0)
        tube_voxels = np.concatenate([tube_voxels, brick_tube_voxels], axis=0)

    if len(stud_voxels) > 0:
        stud_voxels = np.array(stud_voxels, dtype=np.int32)
        stud_voxels = stud_voxels + voxel_offsets
        stud_voxels = stud_voxels // 4
        stud_voxels[:, 1] = -stud_voxels[:, 1]
        mask[stud_voxels[:, 0], stud_voxels[:, 1], stud_voxels[:, 2]] = 1

    if len(tube_voxels) > 0:
        tube_voxels = np.array(tube_voxels, dtype=np.int32)
        tube_voxels = tube_voxels + voxel_offsets
        tube_voxels = tube_voxels // 4
        tube_voxels[:, 1] = -tube_voxels[:, 1]
        mask[tube_voxels[:, 0], tube_voxels[:, 1], tube_voxels[:, 2]] = 2

    return mask
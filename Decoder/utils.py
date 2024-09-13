from Utils.ModelUtils.brick_instance import BrickInstance
import numpy as np


def align_brick_in_voxel_space(brick: BrickInstance, voxel_offsets):
    # Given a brick instance, align it in the voxel space by slightly translating it to its nearest
    # aligned positions.
    voxels = brick.get_brick_voxels()
    translated_voxels = voxels + voxel_offsets
    x_offset = np.min(translated_voxels[:, 0]) % 4
    y_offset = np.min(translated_voxels[:, 1]) % 4
    z_offset = np.min(translated_voxels[:, 2]) % 4
    brick.translate([x_offset, y_offset, z_offset])

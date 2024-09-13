from typing import List
import numpy as np
from Decoder.utils import align_brick_in_voxel_space
from Utils.BrickSet.brick_ids import (
    brick_id_to_class_id,
    brick_id_to_non_reduced_class_id,
    class_id_to_brick_id,
    non_reduced_class_id_to_brick_id,
)
from Utils.GenUtils.generation_util import (
    get_brick_occupancy,
    get_brick_voxel_components,
)
from Utils.GeometryUtils.geometry_utils import get_four_self_rotation
from Utils.ModelUtils.Connection.connpoint import CPoint
from Utils.ModelUtils.Connection.connpointtype import ConnPointType
from Utils.ModelUtils.Connection.conntype import get_conn_type
from Utils.ModelUtils.Connection.cpoints_util import (
    get_trans_matrices,
    modify_cpoints,
    update_ext_conn_point,
)
from Utils.ModelUtils.brick_factory import create_brick
from Utils.ModelUtils.brick_instance import BrickInstance
from tqdm import tqdm


class ReconstructorBase:
    def __init__(
        self,
        ranges: np.ndarray,
        voxel_offsets: np.ndarray,
        has_connection=True,
        is_atomization=True,
    ) -> None:
        self.ranges = ranges
        self.voxel_offsets = voxel_offsets

        self.has_connection = has_connection
        self.is_atomization = is_atomization

        self.iou_threshold = 0.7

        self.semantic_vol: np.ndarray = None
        self.occupancy_map: np.ndarray = None

        self.ext_conn_points: List[CPoint] = []

        if is_atomization:
            self.brick_id_to_class_id = brick_id_to_class_id
            self.class_id_to_brick_id = class_id_to_brick_id
        else:
            self.brick_id_to_class_id = brick_id_to_non_reduced_class_id
            self.class_id_to_brick_id = non_reduced_class_id_to_brick_id

    def init(self, semantic_vol: np.ndarray):
        self.semantic_vol = semantic_vol
        self.occupancy_map = np.zeros_like(semantic_vol, dtype=np.int32)
        self.structure: List[BrickInstance] = []

        assert np.all(np.array(self.semantic_vol.shape) == self.ranges)

    def compute_iou(self, brick_id: str, *voxel_components):
        raise NotImplementedError

    def locate_ground_voxels(self):
        candidate_voxels = np.array(
            np.where((self.semantic_vol > 0) & (self.occupancy_map == 0))
        ).T

        ret = candidate_voxels[candidate_voxels[:, 1] == 0]

        return ret

    def locate_conn_point_voxels(self, conn_point: CPoint):
        """
        Given a connection point, locate the voxels that are around the connection point.
        """

        candidate_voxels = np.array(
            np.where((self.semantic_vol > 0) & (self.occupancy_map == 0))
        ).T

        conn_point_pos = conn_point.pos.astype(np.float32) + self.voxel_offsets

        # Transform the conn_point_pos into the voxel space from the LDR space.
        conn_point_pos[1] = -conn_point_pos[1]
        conn_point_pos /= 4

        ret = candidate_voxels[
            np.linalg.norm(candidate_voxels - conn_point_pos, axis=1) < 3
        ]

        return ret

    def get_potential_brick_ids(self, voxels: np.ndarray):
        raise NotImplementedError

    def arrange_a_brick(self, ground_touched: bool):
        # initialize variables to track the best brick
        best_id = None
        best_iou = -1.0
        best_rot = None
        best_translation = None
        best_transform = None
        found_best = False

        if ground_touched:
            # if the brick should be placed on the ground

            # Get the potential brick ids that can cover the ground voxels.
            voxels = self.locate_ground_voxels()
            potential_brick_ids = self.get_potential_brick_ids(voxels)

            # 1. Enumerate all possible bricks types
            for brick_id in potential_brick_ids:
                brick = create_brick(brick_id, 9, np.identity(4))
                modify_cpoints(brick.template, simplex_reduction=self.is_atomization)

                # 2. Enumerate all possible orientations
                rot_mats = get_four_self_rotation(np.array([0, 1, 0], dtype=np.float32))

                for rot_mat in rot_mats:
                    brick.reset_transformation()
                    brick.rotate(rot_mat)
                    align_brick_in_voxel_space(brick, self.voxel_offsets)

                    brick_occupancy = get_brick_occupancy(brick, self.voxel_offsets)

                    if self.has_connection:
                        voxel_components = get_brick_voxel_components(
                            brick, self.voxel_offsets
                        )
                        voxel_components = list(voxel_components)

                    else:
                        voxel_components = [brick_occupancy]

                    # Translate the brick to the ground level
                    minx, miny, minz = np.min(brick_occupancy, axis=0)
                    maxx, maxy, maxz = np.max(brick_occupancy, axis=0)

                    # let translate_y be 0 - miny
                    ty = -miny

                    # 3. Enumerate all possible translations on the ground level.
                    # Prune bricks exceeding the boundary

                    if miny + ty < 0 or maxy + ty >= self.ranges[1]:
                        continue

                    # iterate the start point along x and z axis
                    for x in range(self.ranges[0] - maxx + minx):
                        tx = -minx + x

                        if minx + tx < 0 or maxx + tx >= self.ranges[0]:
                            continue

                        for z in range(self.ranges[2] - maxz + minz):
                            tz = -minz + z

                            if minz + tz < 0 or maxz + tz >= self.ranges[2]:
                                continue

                            # 4. Check if the brick has collision with the current occupancy map
                            curr_occupancy = brick_occupancy + np.array([tx, ty, tz])
                            if np.any(
                                self.occupancy_map[
                                    curr_occupancy[:, 0],
                                    curr_occupancy[:, 1],
                                    curr_occupancy[:, 2],
                                ]
                                > 0
                            ):
                                continue

                            # Compute the iou of the current brick instance.
                            curr_voxel_components = [
                                vc + np.array([tx, ty, tz]) for vc in voxel_components
                            ]
                            iou = self.compute_iou(brick_id, *curr_voxel_components)

                            # 5. Check if the brick has iou less than the threshold.
                            if iou < self.iou_threshold:
                                continue

                            # 6. Update the best brick.
                            if iou > best_iou:
                                best_iou = iou
                                best_id = brick_id
                                best_rot = rot_mat
                                # Note that the one voxel unit corresponds to 4 ldu.
                                best_translation = np.array([tx, -ty, tz]) * 4

                                # Stop the loop if the best iou is already 1.0
                                if best_iou > 1.0 - 1e-5:
                                    found_best = True
                                    break

                        if found_best:
                            break

                    if found_best:
                        break

                if found_best:
                    break

            # Place the actual brick into the current structure.
            if best_iou > self.iou_threshold:
                self.place_a_brick(best_id, best_rot, best_translation, best_transform)
                return True
            else:
                return False

        else:
            # 1. Enumerate all possible ext conn points from bottom to top
            for cp_old in self.ext_conn_points:

                # Get the potential_brick_ids that can cover the voxels near cp_old
                voxel = self.locate_conn_point_voxels(cp_old)

                if voxel.shape[0] == 0:
                    continue

                potential_brick_ids = self.get_potential_brick_ids(voxel)

                if (
                    cp_old.type == ConnPointType.SIDE3024IN
                    or cp_old.type == ConnPointType.SIDE3024OUT
                ):
                    potential_brick_ids = (
                        ["3024"] if "3024" in potential_brick_ids else []
                    )

                elif (
                    cp_old.type == ConnPointType.SIDE3070IN
                    or cp_old.type == ConnPointType.SIDE3070OUT
                ):
                    potential_brick_ids = (
                        ["3070b"] if "3070b" in potential_brick_ids else []
                    )

                # 2. Enumerate all possible brick types
                for brick_id in potential_brick_ids:
                    brick = create_brick(brick_id, 9, np.identity(4))
                    modify_cpoints(
                        brick.template, simplex_reduction=self.is_atomization
                    )
                    align_brick_in_voxel_space(brick, self.voxel_offsets)

                    # 3. Enumerate all connection points on the brick
                    for cp_new in brick.get_current_conn_points():
                        conn_type = get_conn_type(cp_old, cp_new)
                        if conn_type is None:
                            continue

                        trans_list = get_trans_matrices(cp_old, cp_new)

                        if len(trans_list) == 0:
                            continue

                        # 3. Enumerate all possible placements of the bricks by connecting cp_old and cp_new
                        for mat in trans_list:
                            brick.reset_transformation()
                            align_brick_in_voxel_space(brick, self.voxel_offsets)
                            brick.transform(mat)

                            brick_occupancy = get_brick_occupancy(
                                brick, self.voxel_offsets
                            )

                            # Test if the brick is inside the boundary
                            min_values = np.min(brick_occupancy, axis=0)
                            max_values = np.max(brick_occupancy, axis=0)

                            if np.any(min_values < 0) or np.any(
                                max_values >= self.ranges
                            ):
                                continue

                            # Test if the brick has collision with the current occupancy map
                            if np.any(
                                self.occupancy_map[
                                    brick_occupancy[:, 0],
                                    brick_occupancy[:, 1],
                                    brick_occupancy[:, 2],
                                ]
                                > 0
                            ):
                                continue

                            # Compute the iou of the current brick instance.
                            voxel_components = (
                                get_brick_voxel_components(brick, self.voxel_offsets)
                                if self.has_connection
                                else [brick_occupancy]
                            )

                            iou = self.compute_iou(brick_id, *voxel_components)

                            # Test if the iou is greater than the threshold
                            if iou < self.iou_threshold:
                                continue

                            # Update the best brick
                            if iou > best_iou:
                                best_iou = iou
                                best_id = brick_id
                                best_rot = None
                                best_translation = None
                                best_transform = brick.trans_matrix

                                if best_iou > 1.0 - 1e-5:
                                    found_best = True
                                    break

                        if found_best:
                            break

                    if found_best:
                        break

                if found_best:
                    break

            if best_iou > self.iou_threshold:
                self.place_a_brick(best_id, best_rot, best_translation, best_transform)
                return True
            else:
                return False

    def place_a_brick(self, best_id, best_rot, best_translation, best_transform):
        # Create the best brick
        best_brick = create_brick(best_id, 9, np.identity(4))

        if best_transform is not None:
            best_brick.transform(best_transform)
        else:
            best_brick.rotate(best_rot)
            align_brick_in_voxel_space(best_brick, self.voxel_offsets)
            best_brick.translate(best_translation)

        # Add the best brick into the current structure
        self.structure.append(best_brick)

        # Update the occupancy map
        brick_occupancy = get_brick_occupancy(best_brick, self.voxel_offsets)
        self.occupancy_map[
            brick_occupancy[:, 0], brick_occupancy[:, 1], brick_occupancy[:, 2]
        ] = 1

        # Update the external connectable points
        update_ext_conn_point(best_brick, self.ext_conn_points, None, None)

    def reconstruct(self):
        ground_touched = True

        target_voxel_num = np.sum(self.semantic_vol > 0)

        pbar = tqdm(total=target_voxel_num)

        while True:
            ret = self.arrange_a_brick(ground_touched)
            if ret:
                inc = np.sum(self.occupancy_map > 0) - pbar.n
                pbar.update(n=inc)
            else:
                if ground_touched:
                    ground_touched = False
                else:
                    break

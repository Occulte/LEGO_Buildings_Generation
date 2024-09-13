import base64
import copy
import itertools as iter
import logging

# import open3d as o3d
from io import BytesIO

import numpy as np
from PIL import Image
from numpy import linalg as LA


from Utils.GeometryUtils.cuboid_collision import cub_collision_detect
from Utils.GeometryUtils.geometry_utils import (
    vec_local2world,
    point_local2world,
    group_triangle_to_faces,
)

from Utils.ModelUtils.brick_template import BrickTemplate
from Utils.ModelUtils.unit import Unit
from Utils.ModelUtils.Connection.conntype import (
    ConnType,
    compute_conn_type_by_line_segment,
)
from Utils.ModelUtils.Connection.connpoint import CPoint
from Utils.BrickSet.brick_ids import does_brick_need_augmented_conn_voxel


class BrickInstance:
    ID = 0

    def __init__(self, template: BrickTemplate, trans_matrix, color=15):
        self.template = template
        self.trans_matrix = trans_matrix
        self.color = color
        self.id = BrickInstance.ID

        self.bboxes = None
        self.brick_bbox = None

        BrickInstance.ID += 1

    @staticmethod
    def get_id():
        res = BrickInstance.ID
        BrickInstance.ID += 1
        return res

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BrickInstance) and self.template.id == other.template.id:
            if (
                np.max(self.trans_matrix - other.trans_matrix)
                - np.min(self.trans_matrix - other.trans_matrix)
                < 1e-6
            ):
                return True
            else:
                self_c_points = self.get_current_conn_points()
                other_c_points = other.get_current_conn_points()
                for i in range(len(self_c_points)):
                    if self_c_points[i] not in other_c_points:
                        return False
                return True
        else:
            return False

    def copy(self, deep_copy_transform=True):
        result = copy.copy(self)
        if deep_copy_transform:
            result.trans_matrix = copy.deepcopy(self.trans_matrix)
        return result

    def collide(self, other):
        self_brick_bbox = self.get_brick_bbox()
        other_brick_bbox = other.get_brick_bbox()
        if not cub_collision_detect(self_brick_bbox, other_brick_bbox):
            return False

        self_bbox = self.get_bboxes()
        other_bbox = other.get_bboxes()
        for bb1, bb2 in iter.product(self_bbox, other_bbox):
            if cub_collision_detect(bb1, bb2):
                return True
        return False

    def connect(self, other):
        self_brick_bbox = self.get_enlarged_bbox()
        other_brick_bbox = other.get_enlarged_bbox()
        collided = cub_collision_detect(self_brick_bbox, other_brick_bbox)
        if not collided:
            return None

        conn_points = []
        self_conn_points = self.get_current_conn_points()
        other_conn_points = other.get_current_conn_points()
        for self_idx, other_idx in iter.product(
            range(len(self_conn_points)), range(len(other_conn_points))
        ):
            type = compute_conn_type_by_line_segment(
                self_conn_points[self_idx], other_conn_points[other_idx]
            )
            if type is not None:
                conn_points.append(
                    (type, self_idx, other_idx, self_conn_points[self_idx])
                )

        if len(conn_points) == 0:
            return None
        else:
            return conn_points

    # def block_direction(self, other, block_distance = 15.0, use_cpoint_orientation=False, direction_list=[(1,0,0),...]):
    """      
        The block direction should return a list of boolean for the input direction list.
          1.  For brick in connection, only the direction of connection will be available for removing.
          2.  Also it is necessary to check the colliding for the pair of bricks within block distance.
    """

    def block_direction(
        self, other: "BrickInstance", direction_list, block_distance=15.0, move_steps=1
    ):
        """Test whether self blocks other"""
        other_brick = other.copy()
        dir_vecs = [direction for direction in direction_list]
        dir_blocked = np.full(len(dir_vecs), False, dtype=bool)
        conn_data = self.connect(other_brick)
        if conn_data is not None:
            for conn_type, self_idx, other_idx, self_conn_point in conn_data:
                if conn_type in {
                    ConnType.CLIP_POST,
                    ConnType.CROSSHOLE_POST,
                    ConnType.HOLSTUD_POST,
                }:
                    continue
                orient = self_conn_point.orient
                for idx in range(len(dir_vecs)):
                    # different orientation from the connection orientation
                    if (
                        LA.norm(dir_vecs[idx] - orient) > 1e-2
                        and LA.norm(dir_vecs[idx] + orient) > 1e-2
                    ):
                        dir_blocked[idx] = True
        for step in range(1, move_steps + 1):
            for i in range(len(dir_vecs)):
                dir = dir_vecs[i]
                other_brick.translate(dir * block_distance * step)
                if self.collide(other_brick):
                    dir_blocked[i] = True
                other_brick.translate(-dir * block_distance * step)
        if sum(dir_blocked) == len(dir_vecs):
            dir_blocked = np.full(len(dir_vecs), False, dtype=bool)
        return dir_blocked

    def get_brick_bbox(self):
        if self.brick_bbox is None:
            if self.template.whole_bbox is None:
                # print(f'no bbox for {self.template.id}, force to compute bbox in old way')
                if len(self.get_bbox_vertices()) == 0:
                    self.brick_bbox = {
                        "Origin": None,
                        "Rotation": None,
                        "Dimension": None,
                    }
                else:
                    corner_pos = np.array(self.get_bbox_vertices())
                    max_x = np.amax(corner_pos[:, 0])
                    min_x = np.amin(corner_pos[:, 0])
                    max_y = np.amax(corner_pos[:, 1])
                    min_y = np.amin(corner_pos[:, 1])
                    max_z = np.amax(corner_pos[:, 2])
                    min_z = np.amin(corner_pos[:, 2])
                    origin = [
                        (max_x + min_x) / 2,
                        (max_y + min_y) / 2,
                        (max_z + min_z) / 2,
                    ]
                    dim = [max_x - min_x, max_y - min_y, max_z - min_z]
                    self.brick_bbox = {
                        "Origin": origin,
                        "Rotation": np.identity(3),
                        "Dimension": dim,
                    }
            else:
                vector = np.append(self.template.whole_bbox["Origin"], 1)
                origin = np.transpose(np.dot(self.trans_matrix, np.transpose(vector)))[
                    :3
                ]
                self.brick_bbox = {
                    "Origin": origin,
                    "Rotation": self.get_rotation(),
                    "Dimension": self.template.whole_bbox["Dimension"],
                }
        return self.brick_bbox

    def get_brick_voxels(self):
        if not hasattr(self.template, "voxels") or self.template.voxels.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int32)

        trans_mat = np.round(self.trans_matrix)
        voxels = np.concatenate(
            [
                self.template.voxels,
                np.ones((self.template.voxels.shape[0], 1), dtype=np.int32),
            ],
            axis=1,
        )

        voxels = trans_mat @ voxels.T
        voxels = voxels.T
        voxels = voxels[:, :3]
        voxels = voxels.astype(np.int32)
        return voxels

    def get_stud_tube_voxels(self):
        if does_brick_need_augmented_conn_voxel[self.template.id]:
            stud_voxels_key_name = "stud_voxel_augmented"
            tube_voxels_key_name = "tube_voxel_augmented"
        else:
            stud_voxels_key_name = "stud_voxels"
            tube_voxels_key_name = "tube_voxels"

        # stud_voxels_key_name = "stud_voxels" if not augment else "stud_voxel_augmented"
        # tube_voxels_key_name = "tube_voxels" if not augment else "tube_voxel_augmented"

        if not hasattr(self.template, stud_voxels_key_name) or not hasattr(
            self.template, tube_voxels_key_name
        ):
            return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=np.int32)

        trans_mat = np.round(self.trans_matrix)
        template_stud_voxels = getattr(self.template, stud_voxels_key_name)
        template_tube_voxels = getattr(self.template, tube_voxels_key_name)

        if template_stud_voxels.shape[0] == 0:
            stud_voxels = np.zeros((0, 3), dtype=np.int32)
        else:
            stud_voxels = np.concatenate(
                [
                    template_stud_voxels,
                    np.ones((template_stud_voxels.shape[0], 1), dtype=np.int32),
                ],
                axis=1,
            )
            stud_voxels = trans_mat @ stud_voxels.T
            stud_voxels = stud_voxels.T
            stud_voxels = stud_voxels[:, :3]
            stud_voxels = stud_voxels.astype(np.int32)

        if template_tube_voxels.shape[0] == 0:
            tube_voxels = np.zeros((0, 3), dtype=np.int32)
        else:
            tube_voxels = np.concatenate(
                [
                    template_tube_voxels,
                    np.ones((template_tube_voxels.shape[0], 1), dtype=np.int32),
                ],
                axis=1,
            )
            tube_voxels = trans_mat @ tube_voxels.T
            tube_voxels = tube_voxels.T
            tube_voxels = tube_voxels[:, :3]
            tube_voxels = tube_voxels.astype(np.int32)

        return stud_voxels, tube_voxels

    def get_bbox_vertices(self):
        bbox_ls = self.get_bboxes()
        cub_corner = []
        corner_transform = np.array([[1, 1, 1], [-1, -1, -1]])
        for bbox in bbox_ls:
            cuboid_center = np.array(
                [
                    bbox["Dimension"][0] / 2,
                    bbox["Dimension"][1] / 2,
                    bbox["Dimension"][2] / 2,
                ]
            )
            cuboid_corner_relative = (np.tile(cuboid_center, (2, 1))) * corner_transform
            cub_corners_pos = np.array(
                bbox["Rotation"] @ cuboid_corner_relative.transpose()
            ).transpose() + np.array(bbox["Origin"])
            cub_corner.append(cub_corners_pos[0])
            cub_corner.append(cub_corners_pos[1])
        return cub_corner

    def get_enlarged_bbox(self, enlarge_edge=19):
        bbox = self.get_brick_bbox()
        if bbox["Origin"] is None:
            print(f"No bbox for {self.customize_id} ({self.template.id})")
            return bbox
        else:
            return {
                "Origin": bbox["Origin"],
                "Rotation": bbox["Rotation"],
                "Dimension": [
                    bbox["Dimension"][0] + enlarge_edge,
                    bbox["Dimension"][1] + enlarge_edge,
                    bbox["Dimension"][2] + enlarge_edge,
                ],
            }

    def to_ldraw(self):
        color = self.color
        trans_matrix = self.trans_matrix
        part_id = f"{self.template.id}.dat"
        text = (
            f"1 {color} {trans_matrix[0][3]} {trans_matrix[1][3]} {trans_matrix[2][3]} "
            + f"{trans_matrix[0][0]} {trans_matrix[0][1]} {trans_matrix[0][2]} "
            + f"{trans_matrix[1][0]} {trans_matrix[1][1]} {trans_matrix[1][2]} "
            + f"{trans_matrix[2][0]} {trans_matrix[2][1]} {trans_matrix[2][2]} "
            + f"{part_id}"
        )
        return text

    def rotate(self, rot_mat):
        self.trans_matrix[:3, :3] = np.dot(rot_mat, self.trans_matrix[:3, :3])
        self.bboxes = None
        self.brick_bbox = None

    def translate(self, trans_vec):
        self.trans_matrix[:3, 3:4] = self.trans_matrix[:3, 3:4] + np.reshape(
            trans_vec, (3, 1)
        )
        self.bboxes = None
        self.brick_bbox = None

    def transform(self, trans_mat):
        self.trans_matrix = trans_mat @ self.trans_matrix
        self.bboxes = None
        self.brick_bbox = None

    def get_rotation(self):
        return self.trans_matrix[:3, :3]

    def get_translation(self):
        return self.trans_matrix[:3, 3]

    def reset_transformation(self):
        self.trans_matrix = np.identity(4, dtype=float)
        self.bboxes = None
        self.brick_bbox = None

    def get_translation_for_mesh(self):
        return self.trans_matrix[:3, 3] / 2.5

    def transform_polygon(self, poly):
        return (self.trans_matrix[:3, :3] @ np.asarray(poly).T).T + self.trans_matrix[
            :3, 3
        ]

    def get_current_units(self):
        rot_mat = self.trans_matrix[:3, :3]
        translation = self.trans_matrix[:3, 3]
        return [
            Unit(
                rot_mat @ u.pos + translation,
                rot_mat @ u.orient,
                rot_mat @ u.bi_orient,
                u.shape,
            )
            for u in self.template.units
        ]

    def get_current_conn_points(self):
        conn_points = []

        for cp in self.template.c_points:
            conn_point_orient = vec_local2world(self.trans_matrix[:3, :3], cp.orient)
            conn_point_bi_orient = vec_local2world(
                self.trans_matrix[:3, :3], cp.bi_orient
            )
            conn_point_position = point_local2world(
                self.trans_matrix[:3, :3], self.trans_matrix[:3, 3], cp.pos
            )
            conn_points.append(
                CPoint(
                    conn_point_position,
                    conn_point_orient,
                    conn_point_bi_orient,
                    cp.type,
                    cp.length,
                )
            )

        return conn_points

    def get_bboxes(self):
        if self.bboxes is None:
            bbox = []
            brick_rot = self.get_rotation()
            brick_trans = self.get_translation()
            for box in self.template.bboxes:
                origin = brick_rot @ box["Origin"] + brick_trans
                rotation = brick_rot @ box["Rotation"]
                bbox.append(
                    {
                        "Origin": origin,
                        "Rotation": rotation,
                        "Dimension": box["Dimension"],
                    }
                )

            self.bboxes = bbox

        return self.bboxes

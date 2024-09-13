import numpy as np
from Utils.GeometryUtils.cuboid_collision import cub_collision_detect
from Utils.ModelUtils.Connection.connpointtype import ConnPointType, typeToBrick
from Utils.GeometryUtils.geometry_utils import (
    gen_lateral_vec,
    rot_matrix_from_two_basis,
)


class CPoint:
    def __init__(self, pos, orient, bi_orient, type: ConnPointType, length=0):
        # local position of this connection point
        self.pos: np.ndarray = np.array(pos, dtype=np.float64)
        # local orientation of this connection point
        self.orient: np.ndarray = np.array(orient, dtype=np.float64)
        # axis-aligned vector that perpendicular to the orient
        self.bi_orient: np.ndarray = np.array(bi_orient, dtype=np.float64)
        # the length of the conn point
        self.length = length
        # type (hole, pin, axle, or axle hole)
        self.type = type

    def __eq__(self, other):
        """Overrides the default implementation"""
        if (
            isinstance(other, CPoint)
            and self.type == other.type
            and np.linalg.norm(self.pos - other.pos) < 1
            and np.linalg.norm(self.orient - other.orient) < 1e-3
            and abs(self.length - other.length) < 1e-3
        ):
            return True
        return False

    # NOTE: For visualization only, may not need it.
    def to_ldraw(self) -> str:
        scale_mat = np.identity(3)
        scale_factor = self.length / typeToBrick[self.type][4]
        for i in range(3):
            scale_mat[i][i] *= typeToBrick[self.type][3][i]
            scale_mat[i][i] *= 1 + (scale_factor - 1) * typeToBrick[self.type][1][i]

        rot_mat = self._get_rotation_matrix()
        # rot_mat = rot_matrix_from_vec_a_to_b(typeToBrick[self.type][1], self.orient)
        matrix = rot_mat
        matrix = matrix @ scale_mat
        offset = rot_mat @ (np.array(typeToBrick[self.type][2]) * scale_factor)
        text = (
            f"1 4 {self.pos[0] + offset[0]} {self.pos[1] + offset[1]} {self.pos[2] + offset[2]} "
            + f"{matrix[0][0]} {matrix[0][1]} {matrix[0][2]} "
            + f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]} "
            + f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]} "
            + typeToBrick[self.type][0]
        )
        return text

    def collide(self, other_brick):
        self_box = {
            "Origin": self.pos,
            "Rotation": np.identity(3),
            "Dimension": [2, 2, 2],
        }
        other_bbox = other_brick.get_bboxes()
        for bb in other_bbox:
            if cub_collision_detect(self_box, bb):
                return True
        return False

    def get_line_segment(self):
        p1 = self.pos - self.length / 2 * self.orient
        p2 = self.pos + self.length / 2 * self.orient

        return p1, p2

    def _get_rotation_matrix(self):
        return rot_matrix_from_two_basis(
            typeToBrick[self.type][1],
            gen_lateral_vec(typeToBrick[self.type][1]),
            self.orient,
            self.bi_orient,
        )

import numpy as np
from numpy import linalg as LA

from Utils.GeometryUtils.geometry_utils import (
    gen_lateral_vec,
    rot_matrix_from_two_basis,
)


class Unit:
    ID = 0

    def __init__(self, pos, orient, bi_orient, shape):
        self.id = Unit.ID
        self.pos = pos
        self.orient = orient
        self.bi_orient = bi_orient
        self.shape = shape
        Unit.ID += 1

    def __str__(self):
        unit_dict = {
            "pos": self.pos,
            "orient": self.orient,
            "bi_orient": self.bi_orient,
            "shape": self.shape,
        }
        return str(unit_dict)

    def __eq__(self, other):
        if isinstance(other, Unit):
            if self.shape != other.shape or LA.norm(self.pos - other.pos) > 1e-1:
                return False
            else:
                cos = np.dot(self.orient, other.orient) / (
                    np.linalg.norm(self.orient) * np.linalg.norm(other.orient)
                )
                if abs(cos - 1.0) < 1e-3:
                    return True
                else:
                    return False

    def _get_rotation_matrix(self):
        return rot_matrix_from_two_basis(
            [0, -1, 0], gen_lateral_vec([0, -1, 0]), self.orient, self.bi_orient
        )

    def to_ldraw(self, color=5) -> str:
        rot_mat = self._get_rotation_matrix()
        # rot_mat = rot_matrix_from_vec_a_to_b(typeToBrick[self.type][1], self.orient)
        matrix = rot_mat
        offset = rot_mat @ np.array([0, -4, 0])
        shape_dat = "3024.dat"
        if self.shape == "tile":
            shape_dat = "3070b.dat"
        text = (
            "1 "
            + str(color)
            + f" {self.pos[0] + offset[0]} {self.pos[1] + offset[1]} {self.pos[2] + offset[2]} "
            + f"{matrix[0][0]} {matrix[0][1]} {matrix[0][2]} "
            + f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]} "
            + f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]} "
            + shape_dat
        )
        return text

from Utils.BrickSet.brick_ids import brick_ids
from Utils.GeometryUtils.geometry_utils import (
    get_four_self_rotation,
    rot_matrix_from_two_basis,
)
from Utils.ModelUtils.brick_factory import create_brick

from Utils.ModelUtils.brick_template import BrickTemplate
from Utils.ModelUtils.Connection.connpoint import CPoint
from Utils.ModelUtils.Connection.connpointtype import ConnPointType, stringToType
from Utils.ModelUtils.Connection.conntype import compute_conn_type_by_line_segment
from Utils.ModelUtils.unit import Unit

import json
import os
import numpy as np

from typing import Union

_cpoint_info = {}

_cpoint_info_path = os.path.join("Static", "modify_cpoint_info.json")

with open(_cpoint_info_path, "r") as f:
    _cpoint_info = json.load(f)

_modified_list = set()


def modify_cpoints(brick_template: BrickTemplate, simplex_reduction=True):
    real_id = brick_template.id
    if real_id in _modified_list:
        return

    # Check if the brick is in the modification list
    if (
        real_id not in _cpoint_info["remove"].keys()
        and real_id not in _cpoint_info["add"].keys()
    ):
        return

    if not simplex_reduction and real_id in ["3024", "3070b"]:
        return

    print(f"Modifying brick {real_id}")
    # print(f"Original cpoints:")
    # for cpoint in brick_template.c_points:
    #     print(f"pos: {cpoint.pos}, orient: {cpoint.orient}, type: {cpoint.type}")

    # If to remove a cpoint from this template.
    if real_id in _cpoint_info["remove"].keys():
        removed_cpoints_id = []
        for remove_cpoint in _cpoint_info["remove"][real_id]:
            remove_cpoint = CPoint(
                remove_cpoint["pos"],
                remove_cpoint["orient"],
                remove_cpoint["bi_orient"],
                stringToType[remove_cpoint["type"]],
                remove_cpoint["length"],
            )
            for i, brick_cpoint in enumerate(brick_template.c_points):
                if remove_cpoint == brick_cpoint:
                    removed_cpoints_id.append(i)
                    break

        brick_template.c_points = [
            brick_template.c_points[i]
            for i in range(len(brick_template.c_points))
            if i not in removed_cpoints_id
        ]

    # If to add a cpoint from this template
    if real_id in _cpoint_info["add"].keys():
        add_cpoint_id = []
        for add_cpoint in _cpoint_info["add"][real_id]:
            add_cpoint = CPoint(
                add_cpoint["pos"],
                add_cpoint["orient"],
                add_cpoint["bi_orient"],
                stringToType[add_cpoint["type"]],
                add_cpoint["length"],
            )
            add_cpoint_id.append(add_cpoint)

        brick_template.c_points.extend(add_cpoint_id)

    # print(f"Modified cpoints:")
    # for cpoint in brick_template.c_points:
    #     print(f"pos: {cpoint.pos}, orient: {cpoint.orient}, type: {cpoint.type}")

    _modified_list.add(real_id)


def get_trans_matrices(base_obj: Union[CPoint, Unit], align_obj: Union[CPoint, Unit]):
    transformations = []
    orient_matrice = rot_matrix_from_two_basis(
        align_obj.orient,
        align_obj.bi_orient,
        base_obj.orient,
        base_obj.bi_orient,
    )

    # if the connection is virtual conncetion points for merging, there is no need for rotation
    rotations = get_four_self_rotation(base_obj.orient)

    if isinstance(base_obj, CPoint) and (
        base_obj.type == ConnPointType.SIDE3024IN
        or base_obj.type == ConnPointType.SIDE3024OUT
        or base_obj.type == ConnPointType.SIDE3070IN
        or base_obj.type == ConnPointType.SIDE3070OUT
    ):
        rotations = rotations[:1]

    for orient_rotate_mat in rotations:
        transformation = np.identity(4)
        transform_mat = orient_rotate_mat @ orient_matrice
        new_align_pos = transform_mat @ align_obj.pos
        dis = base_obj.pos - new_align_pos
        transformation[:3, 3] = dis
        transformation[:3, :3] = transform_mat
        transformations.append(transformation)

    return transformations


def update_ext_conn_point(
    new_brick, old_ext_conn_points, old_conn_point, new_conn_point
):
    """
    Update the external connectable points listed in old_ext_conn_points after adding a new brick `new_brick`.
    `new_brick` is the new brick to be added.
    `old_ext_conn_points` is the list of external connectable points before adding `new_brick`.
    `old_conn_point` is one *known* conn point on the old bricks that is connected to the `new_brick`.
    `new_conn_point` is one *known* conn point on the `new_brick` that is connected to the old bricks.
    ---

    If old_conn_point and new_conn_point are known, we can find other pairs of connections points between
    the new brick and the old bricks by setting the known conn points as references.

    Otherwise we need to find all the connected points between the new brick and the old bricks using two for loops.

    """
    # print(f"number of ext conn points before updating: {len(old_ext_conn_points)}")

    old_ext_conn_points.extend(new_brick.get_current_conn_points())

    additional_cpoints_tobe_removed = []

    # if the pair of connection points is not known, find the connection points
    if old_conn_point is None or new_conn_point is None:
        for c_p in new_brick.get_current_conn_points():
            for c_p_old in old_ext_conn_points:
                if compute_conn_type_by_line_segment(c_p, c_p_old) is not None:
                    additional_cpoints_tobe_removed.append(c_p_old)
                    additional_cpoints_tobe_removed.append(c_p)

    # since we only provide one pair of connection points, we need to find all the connected points between two bricks
    else:
        for c_p in new_brick.get_current_conn_points():
            pos_offset = c_p.pos - new_conn_point.pos
            for c_p_old in old_ext_conn_points:
                if np.linalg.norm(c_p_old.pos - old_conn_point.pos - pos_offset) < 1e-1:
                    if compute_conn_type_by_line_segment(c_p, c_p_old) is not None:
                        additional_cpoints_tobe_removed.append(c_p_old)
                        additional_cpoints_tobe_removed.append(c_p)

    # Remove the additional connection points
    # But do not create a new list using list comprehension
    for remove_cp in additional_cpoints_tobe_removed:
        old_ext_conn_points.remove(remove_cp)

    # print(f"number of ext conn points after updating: {len(old_ext_conn_points)}")


if __name__ == "__main__":
    # Test what the hell is query.identify_real_brick_id
    for brick_id in brick_ids:
        brick = create_brick(brick_id)
        connPoints = brick.get_current_conn_points()
        for cp in connPoints:
            if (
                cp.type == ConnPointType.STUD_HOLLOW
                or cp.type == ConnPointType.CIRCLE_TUBE
            ):
                print(f"brick {brick_id} has {cp.type}")

import numpy as np
from numpy.linalg import norm


def check_ineq(T, L, cub_vec, ref_vec):
    right = (
        abs(cub_vec[0].dot(L))
        + abs(cub_vec[1].dot(L))
        + abs(cub_vec[2].dot(L))
        + abs(ref_vec[0].dot(L))
        + abs(ref_vec[1].dot(L))
        + abs(ref_vec[2].dot(L))
    )
    return abs(T.dot(L)) >= right


def get_edge_axis(projection_axis, cub_corners_pos):
    """
    This function basically returns
    rotated x-y-z axes as the local axes
    """
    local_axis = []
    for i in range(1, 4):
        x = cub_corners_pos[0] - cub_corners_pos[i]
        x = x / np.linalg.norm(x)
        projection_axis.append(x)
        local_axis.append(x)
    return projection_axis, local_axis


def cub_collision_detect(cuboid_ref, cuboid):
    if (
        cuboid_ref["Origin"] is None or cuboid["Origin"] is None
    ):  # one of the bounding box is None, no collision
        return False

    center_distance = norm(np.array(cuboid_ref["Origin"]) - np.array(cuboid["Origin"]))
    if center_distance > norm(np.array(cuboid_ref["Dimension"]) / 2) + norm(
        np.array(cuboid["Dimension"]) / 2
    ):
        return False

    corner_transform = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    projection_axis = []
    rotation_ref = cuboid_ref["Rotation"]
    rotation_cub = cuboid["Rotation"]
    center_AB_vec = np.array(cuboid_ref["Origin"]) - np.array(cuboid["Origin"])

    # cuboid_corner_relative: [4 x 3], containing four vectors from the cube center to the four corners defined by `corner_transform`.
    cuboid_dis = np.array(
        [
            cuboid["Dimension"][0] / 2,
            cuboid["Dimension"][1] / 2,
            cuboid["Dimension"][2] / 2,
        ]
    )
    cuboid_corner_relative = (np.tile(cuboid_dis, (4, 1))) * corner_transform
    ref_dis = np.array(
        [
            cuboid_ref["Dimension"][0] / 2,
            cuboid_ref["Dimension"][1] / 2,
            cuboid_ref["Dimension"][2] / 2,
        ]
    )
    ref_corner_relative = (np.tile(ref_dis, (4, 1))) * corner_transform

    # The four corners of the cuboid after rotation
    ref_corners_pos = np.array(
        rotation_ref @ ref_corner_relative.transpose()
    ).transpose() + np.array(cuboid_ref["Origin"])
    cub_corners_pos = np.array(
        rotation_cub @ cuboid_corner_relative.transpose()
    ).transpose() + np.array(cuboid["Origin"])

    # Get the local axes of the cuboid (origin at the first corner, and vectors from origin to other three corners as local axes)
    # projection axis is the combination of A_local_axis and B_local_axis
    projection_axis, A_local_axis = get_edge_axis(projection_axis, cub_corners_pos)
    projection_axis, B_local_axis = get_edge_axis(projection_axis, ref_corners_pos)

    cub_vec = A_local_axis * np.array([cuboid_dis]).transpose()
    ref_vec = B_local_axis * np.array([ref_dis]).transpose()

    # print(A_local_axis)
    # print(B_local_axis)
    # print(cub_vec)
    # print(ref_vec)

    for axis in projection_axis:
        if check_ineq(center_AB_vec, axis, cub_vec, ref_vec):
            return False

    for Aedge in A_local_axis:
        for Bedge in B_local_axis:
            cross = np.cross(Aedge, Bedge)
            if norm(cross) < 1e-7:
                cross = np.cross(Aedge, center_AB_vec)
                if norm(cross) < 1e-7:
                    continue
            if check_ineq(center_AB_vec, cross, cub_vec, ref_vec):
                return False
    return True


def get_homogenous_matrix(cuboid):
    scaling = np.diag(cuboid["Dimension"]).astype(np.float32) / 2
    translation = np.array(cuboid["Origin"])
    rotation = np.array(cuboid["Rotation"])

    homogenous_matrix = np.identity(4)
    homogenous_matrix[:3, :3] = rotation @ scaling
    homogenous_matrix[:3, 3] = translation
    return homogenous_matrix


def get_inverse_homogenous_matrix(cuboid):
    inverse_rotation = np.array(cuboid["Rotation"]).T
    inverse_scaling = np.diag(1 / np.array(cuboid["Dimension"]) * 2)
    inverse_mat = np.identity(4)
    inverse_mat[:3, :3] = inverse_scaling @ inverse_rotation
    translation = np.array(cuboid["Origin"])
    inverse_mat[:3, 3] = -inverse_mat[:3, :3] @ translation
    return inverse_mat


def cub_contains_other(cuboid, other):
    cuboid_mat = get_homogenous_matrix(cuboid)
    other_mat = get_homogenous_matrix(other)
    inverse_cuboid_mat = get_inverse_homogenous_matrix(cuboid)
    other_in_cuboid = inverse_cuboid_mat @ other_mat

    other_corners = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, -1, 1, 1],
            [1, -1, -1, 1],
            [-1, 1, 1, 1],
            [-1, 1, -1, 1],
            [-1, -1, 1, 1],
            [-1, -1, -1, 1],
        ]
    ).T
    unit_other_corners = other_in_cuboid @ other_corners
    return np.all(unit_other_corners <= 1) and np.all(unit_other_corners >= -1)

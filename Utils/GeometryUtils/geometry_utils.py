import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R


# return a axis-aligned unit vector that perpendicular to the input_images normal
def gen_lateral_vec(vec: np.ndarray):
    norm = vec / np.linalg.norm(vec)
    result_vec = np.array([0, 0, 0])
    for i in range(3):  # every coordinate in 3 dimension
        result_vec[i] = 1
        if (
            abs(np.linalg.norm(np.cross(result_vec, norm)) - 1) < 1e-6
        ):  # two vectors perpendicular
            return result_vec
        result_vec[i] = 0


def rot_matrix_from_two_basis(a1, a2, b1, b2):
    # Rotation from (a1, a2) to (b1, b2)
    try:
        assert abs(a1 @ a2) < 1e-4 and abs(b1 @ b2) < 1e-4
        assert (
            abs(np.linalg.norm(a1) - 1) < 1e-4
            and abs(np.linalg.norm(a2) - 1) < 1e-4
            and abs(np.linalg.norm(b1) - 1) < 1e-4
            and abs(np.linalg.norm(b2) - 1) < 1e-4
        )
        a3 = np.cross(a1, a2)
        b3 = np.cross(b1, b2)
        X_before = np.empty([3, 3])
        X_before[:, 0] = a1
        X_before[:, 1] = a2
        X_before[:, 2] = a3

        X_after = np.empty([3, 3])
        X_after[:, 0] = b1
        X_after[:, 1] = b2
        X_after[:, 2] = b3
    except:
        print("invalid rotation axis!!!")
        return np.identity(3)
    return X_after @ np.linalg.inv(X_before)


def vec_local2world(rot_mat: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
    return np.dot(rot_mat, local_vec)


def point_local2world(
    rot_mat: np.ndarray, translation: np.ndarray, local_point: np.ndarray
) -> np.ndarray:
    return np.dot(rot_mat, local_point) + translation


def closestDistanceBetweenLines(
    a0,
    a1,
    b0,
    b1,
    clampAll=False,
    clampA0=False,
    clampA1=False,
    clampB0=False,
    clampB1=False,
):
    """Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance
    """

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = b0 - a0
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)


def is_same_normal(tri, tri2):
    tri_list = np.array(tri[:3])
    tri2_list = np.array(tri2[:3])

    v11 = tri_list[1] - tri_list[0]
    v12 = tri_list[2] - tri_list[0]

    v21 = tri2_list[1] - tri2_list[0]
    v22 = tri2_list[2] - tri2_list[0]

    normal_1 = np.cross(v11, v12)
    normal_2 = np.cross(v21, v22)
    # print(normal_1)
    # print(normal_2)

    return np.linalg.norm(np.cross(normal_1, normal_2)) < 1e-5


def compatible(tri, tri2):
    tri_list = tri[:3]
    tri2_list = tri2[:3]

    is_edge_connected = len(set.intersection(set(tri_list), set(tri2_list))) >= 2
    is_same_norm = is_same_normal(tri, tri2)

    return is_edge_connected and is_same_norm


def group_triangle_to_faces(triangle_list):
    assert len(triangle_list) >= 2

    G = nx.Graph()
    G.add_nodes_from([i for i in range(len(triangle_list))])
    for pair in list(itertools.combinations([i for i in range(len(triangle_list))], 2)):
        if compatible(triangle_list[pair[0]], triangle_list[pair[1]]):
            G.add_edge(pair[0], pair[1])

    components = [c for c in nx.connected_components(G)]

    return [[triangle_list[b] for b in c] for c in components]


def get_perpendicular_vec(vec: np.ndarray) -> np.ndarray:
    assert np.linalg.norm(vec) > 0
    perp_vec = None
    if abs(vec[0]) > 1e-10:
        perp_vec = np.array([(-vec[1] - vec[2]) / vec[0], 1, 1])
    elif abs(vec[1]) > 1e-10:
        perp_vec = np.array([1, (-vec[0] - vec[2]) / vec[1], 1])
    else:
        perp_vec = np.array([1, 1, (-vec[0] - vec[1]) / vec[2]])

    return perp_vec / np.linalg.norm(perp_vec)


def add_vertices_and_triangles_from_voxel(
    voxel: np.ndarray,
    vertices: List[Tuple[int, int, int]],
    faces: List[Tuple[int, int, int]],
):
    x, y, z = voxel
    y = -y  # Invert y-axis to match the coordinate system of LEGO LDR
    vi = len(vertices)
    vertices.extend(
        [
            (x, y, z),
            (x, y, z + 1),
            (x, y + 1, z),
            (x, y + 1, z + 1),
            (x + 1, y, z),
            (x + 1, y, z + 1),
            (x + 1, y + 1, z),
            (x + 1, y + 1, z + 1),
        ]
    )

    faces_idx = [
        (0, 1, 2),
        (1, 3, 2),  # Front face
        (4, 6, 5),
        (5, 6, 7),  # Back face
        (0, 2, 4),
        (2, 6, 4),  # Left face
        (1, 5, 3),
        (3, 5, 7),  # Right face
        (0, 4, 1),
        (1, 4, 5),  # Bottom face
        (2, 3, 6),
        (3, 7, 6),  # Top face
    ]

    added_faces = [
        (
            vi + i,
            vi + j,
            vi + k,
        )
        for i, j, k in faces_idx
    ]

    faces.extend(added_faces)


def get_four_self_rotation(orient):
    assert abs(np.linalg.norm(orient) - 1) < 1e-3
    rotations = []

    for i in {0, 1, 2, 3}:
        r = R.from_rotvec(np.pi / 2 * i * orient)
        rotations.append(r.as_matrix())

    return rotations

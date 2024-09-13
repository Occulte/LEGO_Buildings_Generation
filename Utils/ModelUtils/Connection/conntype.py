import enum

import numpy as np

from Utils.ModelUtils.Connection.connpointtype import ConnPointType, isDoubleOriented
from Utils.GeometryUtils.geometry_utils import closestDistanceBetweenLines


class ConnType(enum.Enum):
    HOLE_PIN = 1  # to insert a pin into a circular hole
    HOLE_AXLE = 2  # to insert an axle into a circular hole
    CROSS_AXLE = 3  # to insert an axle into a cross-shaped hole
    BLOCK = 4  # movement constraint bt inter-blocking
    STUD_TUBE = 5  # to insert a stud on a regular brick into a tube
    PRISMATIC = 6
    HOLSTUD_TUBE = 7
    HOLSTUD_POST = 8
    CLIP_POST = 9
    CROSSHOLE_POST = 10
    CIRCLETUBE_STUDCENTER = 11
    STUD_HOLE = 12
    HOLSTUD_HOLE = 13
    BALL_SOCKET = 14
    SIDE_3024 = 15
    SIDE_3070 = 16


conn_type = {
    (ConnPointType.PIN, ConnPointType.HOLE): ConnType.HOLE_PIN,
    (ConnPointType.AXLE, ConnPointType.CROSS_HOLE): ConnType.CROSS_AXLE,
    (ConnPointType.AXLE, ConnPointType.HOLE): ConnType.HOLE_AXLE,
    (ConnPointType.TUBE, ConnPointType.STUD): ConnType.STUD_TUBE,
    (ConnPointType.POST, ConnPointType.STUD_HOLLOW): ConnType.HOLSTUD_POST,
    (ConnPointType.TUBE, ConnPointType.STUD_HOLLOW): ConnType.STUD_TUBE,
    (ConnPointType.POST, ConnPointType.CLIP): ConnType.CLIP_POST,
    (ConnPointType.HOLE, ConnPointType.PIN): ConnType.HOLE_PIN,
    (ConnPointType.CROSS_HOLE, ConnPointType.AXLE): ConnType.CROSS_AXLE,
    (ConnPointType.HOLE, ConnPointType.AXLE): ConnType.HOLE_AXLE,
    (ConnPointType.STUD, ConnPointType.TUBE): ConnType.STUD_TUBE,
    (ConnPointType.STUD_HOLLOW, ConnPointType.POST): ConnType.HOLSTUD_POST,
    (ConnPointType.STUD_HOLLOW, ConnPointType.TUBE): ConnType.STUD_TUBE,
    (ConnPointType.CLIP, ConnPointType.POST): ConnType.CLIP_POST,
    (ConnPointType.CROSS_HOLE, ConnPointType.POST): ConnType.CROSSHOLE_POST,
    (ConnPointType.POST, ConnPointType.CROSS_HOLE): ConnType.CROSSHOLE_POST,
    (
        ConnPointType.CENTER_FOUR_STUD,
        ConnPointType.CIRCLE_TUBE,
    ): ConnType.CIRCLETUBE_STUDCENTER,
    (
        ConnPointType.CIRCLE_TUBE,
        ConnPointType.CENTER_FOUR_STUD,
    ): ConnType.CIRCLETUBE_STUDCENTER,
    (
        ConnPointType.STUD_HOLLOW,
        ConnPointType.CIRCLE_TUBE,
    ): ConnType.CIRCLETUBE_STUDCENTER,
    (
        ConnPointType.CIRCLE_TUBE,
        ConnPointType.STUD_HOLLOW,
    ): ConnType.CIRCLETUBE_STUDCENTER,
    (ConnPointType.STUD, ConnPointType.CIRCLE_TUBE): ConnType.CIRCLETUBE_STUDCENTER,
    (ConnPointType.CIRCLE_TUBE, ConnPointType.STUD): ConnType.CIRCLETUBE_STUDCENTER,
    (ConnPointType.STUD, ConnPointType.HOLE): ConnType.STUD_HOLE,
    (ConnPointType.HOLE, ConnPointType.STUD): ConnType.STUD_HOLE,
    (ConnPointType.STUD_HOLLOW, ConnPointType.HOLE): ConnType.HOLSTUD_HOLE,
    (ConnPointType.HOLE, ConnPointType.STUD_HOLLOW): ConnType.HOLSTUD_HOLE,
    (ConnPointType.BALL, ConnPointType.SOCKET): ConnType.BALL_SOCKET,
    (ConnPointType.SOCKET, ConnPointType.BALL): ConnType.BALL_SOCKET,
    (ConnPointType.SIDE3024OUT, ConnPointType.SIDE3024IN): ConnType.SIDE_3024,
    (ConnPointType.SIDE3024IN, ConnPointType.SIDE3024OUT): ConnType.SIDE_3024,
    (ConnPointType.SIDE3070OUT, ConnPointType.SIDE3070IN): ConnType.SIDE_3070,
    (ConnPointType.SIDE3070IN, ConnPointType.SIDE3070OUT): ConnType.SIDE_3070,
}


def get_conn_type(c_point1, c_point2, err_msg=False):
    if (c_point1.type, c_point2.type) in conn_type.keys():
        return conn_type[(c_point1.type, c_point2.type)]
    else:
        if err_msg:
            print("unsupported connection type!!!")
            print(c_point1.type.name, c_point2.type.name)
            print(c_point1.to_ldraw())
            print(c_point2.to_ldraw())
        return None


def compute_conn_type_by_line_segment(c_point1, c_point2):
    cp11, cp12 = c_point1.get_line_segment()
    cp21, cp22 = c_point2.get_line_segment()
    pA, pB, dist = closestDistanceBetweenLines(cp11, cp12, cp21, cp22, clampAll=True)
    if dist < 1.0 and np.linalg.norm(np.cross(c_point1.orient, c_point2.orient)) < 1e-2:
        if not np.linalg.norm(c_point1.orient - c_point2.orient) < 5e-1 and not (
            isDoubleOriented[c_point1.type] or isDoubleOriented[c_point2.type]
        ):
            return None
        return get_conn_type(c_point1, c_point2)
    else:
        return None  # not connected

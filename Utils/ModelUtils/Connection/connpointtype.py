import enum


class ConnPointType(enum.Enum):
    HOLE = 1
    PIN = 2
    AXLE = 3
    CROSS_HOLE = 4
    SOLID = 5
    STUD = 6
    TUBE = 7
    CIRCLE_TUBE = 8
    CENTER_FOUR_STUD = 9
    STUD_HOLLOW = 13  # see the side face of 87087
    POST = 14  # see the post stickers in the bottom of tiles
    CLIP = 15
    BALL = 16  # classic lego 5.9mm ball
    SOCKET = 17  # paired with BALL type
    SIDE3024OUT = 18  # side face of 3024
    SIDE3024IN = 19  # side face of 3024
    SIDE3070OUT = 20
    SIDE3070IN = 21


stringToType = {
    "hole": ConnPointType.HOLE,
    "pin": ConnPointType.PIN,
    "axle": ConnPointType.AXLE,
    "cross_hole": ConnPointType.CROSS_HOLE,
    "solid": ConnPointType.SOLID,
    "stud": ConnPointType.STUD,
    "tube": ConnPointType.TUBE,
    "circle_tube": ConnPointType.CIRCLE_TUBE,
    "center_four_stud": ConnPointType.CENTER_FOUR_STUD,
    "hollow_stud": ConnPointType.STUD_HOLLOW,
    "post": ConnPointType.POST,
    "clip": ConnPointType.CLIP,
    "ball": ConnPointType.BALL,
    "socket": ConnPointType.SOCKET,
    "side3024out": ConnPointType.SIDE3024OUT,
    "side3024in": ConnPointType.SIDE3024IN,
    "side3070out": ConnPointType.SIDE3070OUT,
    "side3070in": ConnPointType.SIDE3070IN,
}

typeToString = {s: t for t, s in stringToType.items()}

# type to if the connection point is valid in both sides along the normal
isDoubleOriented = {
    ConnPointType.HOLE: True,
    ConnPointType.PIN: False,
    ConnPointType.AXLE: True,
    ConnPointType.CROSS_HOLE: True,
    ConnPointType.SOLID: True,
    ConnPointType.STUD: False,
    ConnPointType.TUBE: False,
    ConnPointType.POST: False,
    ConnPointType.CLIP: True,
    ConnPointType.STUD_HOLLOW: False,
    ConnPointType.CIRCLE_TUBE: False,
    ConnPointType.CENTER_FOUR_STUD: False,
    ConnPointType.BALL: False,
    ConnPointType.SOCKET: False,
    ConnPointType.SIDE3024OUT: False,
    ConnPointType.SIDE3024IN: False,
    ConnPointType.SIDE3070OUT: False,
    ConnPointType.SIDE3070IN: False,
}

# these properties are for visualization ONLY
# property: (debug brick, orientation in local coordinate, offset of the center, scaling in three directions, the length of the debuging model along the orientation direction)
typeToBrick = {
    ConnPointType.HOLE: ("18654.dat", [0, 1, 0], [0, 0, 0], [1, 1, 1], 20),
    ConnPointType.PIN: ("4274.dat", [1, 0, 0], [10, 0, 0], [1, 1, 1], 20),
    ConnPointType.AXLE: ("3704.dat", [1, 0, 0], [0, 0, 0], [0.5, 1, 1], 4),
    ConnPointType.CROSS_HOLE: ("axle.dat", [0, 1, 0], [0, -10, 0], [1, 20, 1], 4),
    ConnPointType.SOLID: (
        "99948.dat",
        [0, 1, 0],
        [0, 0, 0],
        [0.2225, 0.2225, 0.2225],
        1,
    ),
    ConnPointType.STUD: ("stud.dat", [0, 1, 0], [0, 2, 0], [1, 1, 1], 4),
    ConnPointType.STUD_HOLLOW: ("stud2a.dat", [0, 1, 0], [0, 2, 0], [1, 1, 1], 4),
    ConnPointType.TUBE: ("box5.dat", [0, 1, 0], [0, 0.5, 0], [6, 1, 6], 2),
    ConnPointType.POST: ("stud3.dat", [0, 1, 0], [0, 2, 0], [1, 1, 1], 4),
    ConnPointType.CLIP: ("clip3.dat", [1, 0, 0], [0, -2, 0], [1, 1, 1], 4),
    ConnPointType.CENTER_FOUR_STUD: (
        r"8\4-4cyli.dat",
        [0, 1, 0],
        [0, -2, 0],
        [7, 4, 7],
        4,
    ),
    ConnPointType.CIRCLE_TUBE: ("stud4.dat", [0, 1, 0], [0, 2, 0], [1, 1, 1], 2),
    ConnPointType.BALL: (
        "99948.dat",
        [0, 1, 0],
        [0, 0, 0],
        [0.2225, 0.2225, 0.2225],
        1,
    ),
    ConnPointType.SOCKET: (
        "99948.dat",
        [0, 1, 0],
        [0, 0, 0],
        [0.2225, 0.2225, 0.2225],
        1,
    ),
}

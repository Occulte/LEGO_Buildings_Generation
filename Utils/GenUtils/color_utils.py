from typing import Dict
import numpy as np



def color_palette_paper():
    """
    The ultimately correct color palette that should be used in the main paper.
    This should not allow color collision.
    """
    colors: Dict[int, str] = {
        1: "#B77F70",
        2: "#FED59E",
        3: "#BEB1A8",
        4: "#A79A89",
        5: "#8A95A9",
        6: "#F4C7B0",
        7: "#1F2C1C",
        8: "#CDDAA5",
        9: "#FEC37D",
        10: "#D3AC73",
        11: "#7D7465",
        12: "#9FABB9",
        13: "#B0B1B6",
        14: "#99857E",
        15: "#88878D",
        16: "#91A0A5",
        17: "#9AA690",
        18: "#686789",
        19: "#F2EEED",
        20: "#E5E2B9",
        21: "#798223",
        22: "#EFD3AC",
        23: "#75809C",
        24: "#ECCED0",
        25: "#B57C82",
        26: "#E8D3C0",
        27: "#7A8A71",
        28: "#789798",
        29: "#B3CAD8",
        30: "#D6E1D7",
    }
    return {
        key: np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
        / 255.0
        for key, color in colors.items()
    }

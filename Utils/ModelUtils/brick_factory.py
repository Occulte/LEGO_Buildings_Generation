import json
import os.path
import zipfile

import numpy as np
from typing import List, Dict, Tuple, Optional
from Utils.ModelUtils.brick_template import BrickTemplate
from Utils.ModelUtils.brick_instance import BrickInstance

from Utils.ModelUtils.Connection.connpoint import CPoint
from Utils.GeometryUtils.geometry_utils import gen_lateral_vec, get_perpendicular_vec
from Utils.ModelUtils.unit import Unit
from Utils.ModelUtils.Connection.connpointtype import typeToBrick, stringToType
from Utils.BrickSet.brick_ids import brick_ids

_Loaded_templates = {}
local_brick_data_path = (
    r"brick_data.zip"
)


def create_brick(
    brick_id: str, color_id: int = 16, trans_mat: np.ndarray = None
) -> BrickInstance:
    if trans_mat is None:
        trans_mat = np.identity(4)

    return BrickInstance(get_template(brick_id), trans_mat, color_id)


def get_template(brick_id: str) -> BrickTemplate:
    brick_template = _Loaded_templates.get(brick_id, None)
    if brick_template is not None:
        return brick_template
    brick_template = BrickTemplate(
        get_cpoints(brick_id),
        get_bboxes(brick_id),
        brick_id,
    )
    units, brick_type = get_units(brick_id)
    if units is not None:
        brick_template.units = units
    whole_bbox = get_whole_bbox(brick_id)
    if whole_bbox is not None:
        brick_template.whole_bbox = whole_bbox

    _Loaded_templates[brick_id] = brick_template
    return brick_template


def get_bboxes(brick_id: str) -> List[Dict[str, np.ndarray]]:
    filename = f"{brick_id}.col"
    try:
        with zipfile.ZipFile(local_brick_data_path, "r") as zip_ref:
            with zip_ref.open(os.path.join("col", filename), "r") as file:
                col_content = file.read().decode("utf-8")
                result = []
                for line in col_content.strip().splitlines():
                    result.append(bbox_from_str(line))
        return result

    except:
        print(f"Cannot find internal bounding box for {brick_id}")
        return []


def bbox_from_str(bbox_str: str):
    float_data = [float(item) for item in bbox_str.split(" ")[:17]]
    orient = np.array(float_data[2:11]).reshape(3, 3)
    origin = np.array(float_data[11:14])
    dim = np.abs(np.array(float_data[14:17]) * 2 + 1)
    return {"Origin": origin, "Rotation": orient, "Dimension": dim}


def bbox_to_str(bbox: Dict):
    origin = bbox["Origin"].tolist()
    orient = bbox["Rotation"].flatten().tolist()
    dim = (np.array(bbox["Dimension"]) - 1) / 2
    float_data = [
        str(item) for sublist in [[9, 0], orient, origin, dim] for item in sublist
    ]
    bbox_str = " ".join(float_data)
    bbox_str += " null"
    return bbox_str


def get_cpoints_from_json(json_file):
    cpoints = []
    for connPoint in json_file["connPoint"]:
        orient = connPoint["orient"]
        # for axis-aligned unit vector
        if any(
            (
                np.allclose(orient, v)
                for v in [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            )
        ):
            bi_orient = gen_lateral_vec(np.array(orient))
        # for any other vectors
        else:
            bi_orient = get_perpendicular_vec(np.array(orient))
        cpoints.append(
            CPoint(
                pos=connPoint["pos"],
                orient=orient,
                bi_orient=bi_orient,
                type=stringToType[connPoint["type"]],
                length=(
                    typeToBrick[stringToType[connPoint["type"]]][4]
                    if "length" not in connPoint
                    else connPoint["length"]
                ),
            )
        )
    return cpoints


def get_cpoints(brick_id: str) -> List[CPoint]:
    """Get CPoint's for brick_id

    Returns:
        All cpoint's of brick_id if remote has the file, empty list if no such file.
    """
    try:
        with zipfile.ZipFile(local_brick_data_path, "r") as zip_ref:
            with zip_ref.open(f"conn/{brick_id}.json", "r") as file:
                conn_dict: Dict = json.load(file)
                cpoints = get_cpoints_from_json(conn_dict)

        return cpoints

    except:
        assert False, f"Cannot find connection points for {brick_id}"


def get_units(brick_id: str) -> Tuple[Optional[List[Unit]], Optional[str]]:
    try:
        with zipfile.ZipFile(local_brick_data_path, "r") as zip_ref:
            with zip_ref.open(f"unit/{brick_id}.json", "r") as file:
                units_dict = json.load(file)

                assert "shape" in units_dict, f"Cannot find shape for {brick_id}"
                bricktype = units_dict["shape"]
                result_units = []
                for unit_dict in units_dict["units"]:
                    bi_orient = gen_lateral_vec(np.array(unit_dict["orient"]))
                    unit = Unit(
                        unit_dict["pos"],
                        unit_dict["orient"],
                        bi_orient,
                        "plate" if bricktype in {"brick", "plate"} else "tile",
                    )
                    result_units.append(unit)

        return result_units, bricktype
    except:
        return None, None


def get_whole_bbox(brick_id: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        with zipfile.ZipFile(local_brick_data_path, "r") as zip_ref:
            with zip_ref.open(f"whole_box/{brick_id}.json", "r") as file:
                result = json.load(file)
                result["Origin"] = np.array(result["Origin"])
                result["Rotation"] = np.array(result["Rotation"])
                result["Dimension"] = np.array(result["Dimension"])

        return result
    except:
        assert False, f"Cannot find units for {brick_id}"


if __name__ == "__main__":
    for brick_id in brick_ids:
        create_brick(brick_id)

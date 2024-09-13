from typing import Dict, List, Tuple
import os
import numpy as np
from Utils.ModelUtils.Connection.connpoint import CPoint
from Utils.ModelUtils.brick_factory import create_brick


def draw_bbox(bbox, i):
    origin = bbox["Origin"]
    rot_mat = bbox["Rotation"]
    scaling = np.identity(3)
    row, col = np.diag_indices(scaling.shape[0])
    scaling[row, col] = np.array(
        [bbox["Dimension"][0] / 2, bbox["Dimension"][1], bbox["Dimension"][2] / 2]
    )
    matrix = rot_mat @ scaling
    offset = matrix @ np.array([0, -0.5, 0])
    text = (
        f"1 {1+i} {origin[0] + offset[0]} {origin[1] + offset[1]} {origin[2] + offset[2]} "
        + f"{matrix[0][0]} {matrix[0][1]} {matrix[0][2]} "
        + f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]} "
        + f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]} "
        + "box5.dat"
    )
    return text


def read_bricks_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    bricks = []

    for line in lines:
        if line.startswith("0 "):
            continue
        elif line.startswith("1 "):
            brick_info = line.strip().split(" ")
            color = int(brick_info[1])
            translation = np.array([float(i) for i in brick_info[2:5]])
            rotation = np.array([float(i) for i in brick_info[5:14]]).reshape(3, 3)
            brick_id = brick_info[-1].split(".")[0]
            transform_matrix = np.identity(4)
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = translation
            brick = create_brick(brick_id, color, transform_matrix)
            bricks.append(brick)

    return bricks


def write_bricks_to_file(
    bricks: List,
    file_path,
    debug_col_bboxes=False,
    debug_whole_box=False,
    debug_cpoints=False,
    debug_units=False,
    cpoints=[],
    additional_bboxes: List[Tuple[Dict, int]] = [],  # (bbox, color_id)
    additional_cpoints: List[CPoint] = [],
):

    file = open(file_path, "w")

    ldr_content = ""

    if debug_col_bboxes:
        ldr_content = ldr_content + "\n"
        for brick in bricks:
            bbox = brick.get_bboxes()
            ldr_content += "\n".join([draw_bbox(bbox[i], i) for i in range(len(bbox))])
            ldr_content += "\n"
            print(f"curr_ldr_content: {ldr_content}")

    ldr_content += "\n0 STEP \n"

    ldr_content += "\n0 STEP\n".join([brick.to_ldraw() for brick in bricks])

    if debug_cpoints:  # output the connection points
        conn_point = "\n0 STEP\n".join(
            [c.to_ldraw() for brick in bricks for c in brick.get_current_conn_points()]
        )
        ldr_content = ldr_content + "\n" + conn_point

    if debug_units:
        conn_point = "\n0 STEP\n".join(
            [
                u.to_ldraw()
                for brick in bricks
                if hasattr(brick.template, "units")
                for u in brick.get_current_units()
            ]
        )
        ldr_content = ldr_content + "\n" + conn_point

    if debug_whole_box:
        for brick in bricks:
            if brick.get_brick_bbox()["Origin"] is not None:
                ldr_content += f"\n {draw_bbox(brick.get_brick_bbox(), 4)}"
                ldr_content += "\n0 STEP \n"

    for bbox, color_id in additional_bboxes:
        ldr_content += f"\n {draw_bbox(bbox, color_id)}"
        ldr_content += "\n0 STEP \n"

    ldr_content += "\n"
    additional_cpoint_content = "\n0 STEP\n".join(
        [cp.to_ldraw() for cp in additional_cpoints]
    )
    ldr_content += additional_cpoint_content + "\n0 STEP\n"

    ldr_content += "\n"
    ldr_content += "\n0 STEP\n".join([c_p.to_ldraw() for c_p in cpoints])

    file.write(ldr_content)
    file.close()
    # print(f"file {file_path} saved!")


if __name__ == "__main__":
    # Test the read and write of LDR files
    test_file_path = os.path.join("Experiments", "predicted", "new_635.npy_merged.ldr")
    write_file_path = os.path.join(
        "Experiments", "predicted", "new_635.npy_merged_rewritten.ldr"
    )
    bricks = read_bricks_from_file(test_file_path)
    write_bricks_to_file(bricks, write_file_path)

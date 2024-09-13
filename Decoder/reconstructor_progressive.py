from Decoder.reconstructor_base import ReconstructorBase
import numpy as np

from Utils.ModelUtils.IO.io_util import write_bricks_to_file


class ReconstructorProgressive(ReconstructorBase):
    def __init__(
        self,
        ranges: np.ndarray,
        voxel_offsets: np.ndarray,
        has_connection=True,
        is_atomization=True,
    ) -> None:
        super().__init__(ranges, voxel_offsets, has_connection, is_atomization)

    def compute_iou(self, brick_id: str, *voxel_components):
        assert len(voxel_components) == 1 or len(voxel_components) == 3

        if any([vc is None for vc in voxel_components]):
            return 0.0

        try:
            component_ids = (
                [brick_id]
                if len(voxel_components) == 1
                else [brick_id, "stud_voxels", "tube_voxels"]
            )

            iou = 0.0
            for i, vc in enumerate(voxel_components):
                iou += np.sum(
                    self.semantic_vol[vc[:, 0], vc[:, 1], vc[:, 2]]
                    == self.brick_id_to_class_id[component_ids[i]]
                )

            len_voxels = sum([len(vc) for vc in voxel_components])
            iou /= len_voxels if len_voxels > 0 else 0.0

        except Exception as e:
            # print(f"Error in compute_iou: {e}")
            iou = 0.0

        return iou

    def get_potential_brick_ids(self, voxels: np.ndarray):

        class_ids = np.unique(
            self.semantic_vol[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
        )

        potential_brick_ids = [
            self.class_id_to_brick_id[class_id][0]
            for class_id in class_ids
            if class_id in self.class_id_to_brick_id
        ]

        ##################################
        # Prioritize the 2412b brick
        ##################################

        if "2412b" in potential_brick_ids:
            potential_brick_ids.remove("2412b")
            potential_brick_ids.insert(0, "2412b")

        return potential_brick_ids


if __name__ == "__main__":

    import os
    import argparse

    from Decoder.merge import merge_bricks

    parser = argparse.ArgumentParser(description="Reconstruct and merge LEGO bricks.")
    parser.add_argument(
        "--tensor_file",
        type=str,
        help="The filename of the tensor file (should end with .npy).",
    )
    parser.add_argument(
        "--tensor_folder",
        type=str,
        default=".",
        help="The folder containing the tensor file (default is current directory).",
    )
    args = parser.parse_args()


    ranges = np.array([32, 40, 32], dtype=np.int32)
    bounds = {
        "Origin": [0, -100, 0],
        "Rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "Dimension": (ranges * 4).tolist(),
    }
    voxel_offset = ranges * 2 - 2
    voxel_offset[1] = 2

    tensor_file = os.path.join(args.tensor_folder, args.tensor_file)

    reconstructor = ReconstructorProgressive(ranges, voxel_offset)
    reconstructor.init(np.load(tensor_file))
    reconstructor.reconstruct()

    bricks = reconstructor.structure
    merged_bricks, merge_duration = merge_bricks(bricks)

    write_bricks_to_file(
        merged_bricks, tensor_file.replace(".npy", "_reconstruct.ldr")
    )

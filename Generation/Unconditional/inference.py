import os
import torch
import numpy as np
import argparse

from Generation.Unconditional.pipeline import LEGOPipeline

def main(args):
    pipeline = LEGOPipeline.from_pretrained(args.checkpoint_dir, local_files_only=True)
    pipeline.to("cuda", dtype=torch.float16)
    pipeline.unet.eval()
    pipeline.unet.requires_grad = False

    for i in range(args.sample_num // args.batch_size):
        generated_occupancy_maps = pipeline(
            batch_size=args.batch_size,
            num_inference_steps=1000,
        ).occupancy_maps

        save_dir = os.path.join(args.results_dir, "generated_occupancy_maps")
        os.makedirs(save_dir, exist_ok=True)
        for j, occupancy_map in enumerate(generated_occupancy_maps):
            save_path = os.path.join(
                save_dir, f"occupancy_map_{i * args.batch_size + j}.npy"
            )
            np.save(save_path, occupancy_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate occupancy maps using LEGOPipeline."
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where results will be saved.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for inference."
    )
    parser.add_argument(
        "--smaple_num",
        type=int,
        default=1000,
        help="Number of samples to generate.",
    )

    args = parser.parse_args()

    main(args)

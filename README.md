# Learn to Create Simple LEGO Micro Buildings
Office code repository for the paper "Learn to Create Simple LEGO Micro Buildings"

## Dataset

Each data sample includes four files:
- model.ldr
- occupancy_map.npy
- augmented_conn_mask.npy
- combined_tensor.npy
`model.ldr` records the structure for the LEGO model. You can visualize the LEGO assembly with [LDView](https://tcobbs.github.io/ldview/) or [Studio](https://www.bricklink.com/v3/studio/download.page).

`occupancy_map.npy` is a numpy array, recording the semantic volume of the LEGO model.

`augmented_conn_mask.npy` is a numpy array labelling the voxels between multiple studs/tubes for some bricks with multiple studs/tubes.

`combined_tensor.npy` is a numpy array, masking the labels of `occupancy_map.npy` with that of `augmented_conn_mask.npy`.

You can download the dataset via the [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155209932_link_cuhk_edu_hk/Eck2egIgF3hFiWTkVLfEp1MBvgdC4FlWqvSwz4wzgZt2lQ?e=YDAiPY).

## Brick set

You need to download the annotated brick information form the [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155209932_link_cuhk_edu_hk/EUAhfXQBSotOj_3qqvQRt1ABiTlII95Kwv2wtqh0N3Hb3g?e=cWbulc).

Then, you need to replace the `local_brick_data_path` with the download path in the `brick_factory.py`.

## Pre-trained model

You can download the pre-trained model via the [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155209932_link_cuhk_edu_hk/EcYtoSsYeKBFqtgvEvC2AbgBdVuOrQOC_8ssbPPcjxKnvg?e=VKv5RX).

## Train from scratch

If you would like to train the model from scratch, you can run the following command:

```

accelerate launch train_unconditional.py --train_data_dir ${dataset_dir} --output_dir ${output_dir} --resolution 32 40 32 --train_batch_size 20 --eval_batch_size 12 --dataloader_num_workers 20 --validate_epochs 150 --save_model_epochs 150 --num_epochs 2000 --learning_rate 7e-5 --mixed_precision bf16 --ddpm_num_inference_steps 1000 --ddpm_beta_schedule scaled_linear --checkpointing_steps 1000 --checkpoints_total_limit 3 --seed 1443 --prediction_type sample

```

## Inference

Use the following command to run the trained model to generate a semantic volume:

```

accelerate launch inference.py --checkpoint_dir ${checkpoint_dir} --results_dir ${results_dir}

```

## Reconstruct LEGO model

Run the following the command to reconstruct the LEGO model from a semantic volume.

```

python reconstructor_progressive.py --tensor_folder ${tensor_folder} --tensor_file ${tensor_file}

```

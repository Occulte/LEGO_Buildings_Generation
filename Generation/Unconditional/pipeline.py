import torch
import numpy as np

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline


@dataclass
class LEGOPipelineOutput(BaseOutput):
    """
    Output class for LEGO generation pipelines.

    Args:
        occupancy_maps (`np.ndarray`)
            NumPy array of shape `(batch_size, depth, height, width)`.
    """

    occupancy_maps: np.ndarray


class LEGOPipeline(DiffusionPipeline):
    r"""
    Pipeline for LEGO generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet3DModel`]):
            A `UNet3DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, cls_num: int = 1):
        super().__init__()
        self.cls_num = cls_num
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            return_dict: bool = True,
    ) -> Union[LEGOPipelineOutput, Tuple]:
        if isinstance(self.unet.config.sample_size, int):
            occupancy_shape = (
                batch_size,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            occupancy_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            occupancy_map = randn_tensor(occupancy_shape, generator=generator)
            occupancy_map = occupancy_map.to(self.device)
        else:
            occupancy_map = randn_tensor(occupancy_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(occupancy_map, t).sample
            model_output = torch.softmax(model_output, dim=1) * 2 - 1

            # 2. compute previous occupancy_map: x_t -> x_t-1
            occupancy_map = self.scheduler.step(model_output, t, occupancy_map, generator=generator).prev_sample

        # occupancy_map = (occupancy_map / 2 + 0.5).clamp(0, 1)
        # mask_channel = occupancy_map[:, 0, :, :, :]
        # class_channels = occupancy_map[:, 1:, :, :, :]
        # predicted_classes = torch.argmax(class_channels, dim=1)
        # occupancy_map = torch.where(mask_channel > 0.5, predicted_classes + 1, torch.zeros_like(predicted_classes))
        occupancy_map = torch.argmax(occupancy_map, dim=1)

        occupancy_map = occupancy_map.cpu().numpy().astype(np.uint8)

        if not return_dict:
            return (occupancy_map,)

        return LEGOPipelineOutput(occupancy_maps=occupancy_map)


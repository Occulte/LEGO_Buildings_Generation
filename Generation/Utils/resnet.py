from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.activations import get_activation


class Upsample3D(nn.Module):
    """A 3D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 3D layer.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            use_conv_transpose: bool = False,
            out_channels: Optional[int] = None,
            name: str = "conv",
            kernel_size: Optional[int] = None,
            padding=1,
            norm_type=None,
            eps=None,
            elementwise_affine=None,
            bias=True,
            interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        conv_cls = nn.Conv3d

        self.norm = None

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose3d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv3d_0 = conv

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest3d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv3d_0(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 3D layer.
    """

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            padding: int = 1,
            name: str = "conv",
            kernel_size=3,
            norm_type=None,
            eps=None,
            elementwise_affine=None,
            bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        conv_cls = nn.Conv3d

        self.norm = None

        if use_conv:
            conv = conv_cls(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        if name == "conv":
            self.Conv3d_0 = conv
            self.conv = conv
        elif name == "Conv3d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv3d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift"
            for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample3D`] and [`~models.resnet.FirDownsample3D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv3d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_3d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
            self,
            *,
            in_channels: int,
            out_channels: Optional[int] = None,
            conv_shortcut: bool = False,
            dropout: float = 0.0,
            temb_channels: int = 512,
            groups: int = 32,
            groups_out: Optional[int] = None,
            pre_norm: bool = True,
            eps: float = 1e-6,
            non_linearity: str = "swish",
            skip_time_act: bool = False,
            time_embedding_norm: str = "default",  # default, scale_shift,
            kernel: Optional[torch.FloatTensor] = None,
            output_scale_factor: float = 1.0,
            use_in_shortcut: Optional[bool] = None,
            up: bool = False,
            down: bool = False,
            conv_shortcut_bias: bool = True,
            conv_3d_out_channels: Optional[int] = None,
            dilation: bool = False,
    ):
        super().__init__()
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear
        conv_cls = nn.Conv3d

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        if not dilation:
            self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        if not dilation:
            self.conv2 = conv_cls(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = conv_cls(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=2, dilation=2)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample3D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = Downsample3D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
            self,
            input_tensor: torch.FloatTensor,
            temb: torch.FloatTensor,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
                if isinstance(self.upsample, Upsample3D)
                else self.upsample(input_tensor)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
                if isinstance(self.upsample, Upsample3D)
                else self.upsample(hidden_states)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
                if isinstance(self.downsample, Downsample3D)
                else self.downsample(input_tensor)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
                if isinstance(self.downsample, Downsample3D)
                else self.downsample(hidden_states)
            )

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

from typing import Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv3D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: Optional[int] = None,
        num_convs: int = 2,
    ):
        super(BasicConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels if mid_channels is not None else out_channels
        self.num_convs = num_convs
        self._build_layers()

    def _build_layers(self):
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_convs):
            inc = self.in_channels if i == 0 else self.mid_channels
            outc = self.out_channels if i == self.num_convs - 1 else self.mid_channels
            self.conv_layers.append(nn.Conv3d(inc, outc, kernel_size=3, stride=1, padding=1))
            self.conv_layers.append(nn.BatchNorm3d(outc))
            self.conv_layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*self.conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class DownLayer3D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_convs: int = 2,
        pooling_stride: int = 2,
    ):
        super(DownLayer3D, self).__init__()

        self.conv = BasicConv3D(
            in_channels=in_channels, 
            out_channels=out_channels, 
            num_convs=num_convs
        )
        self.pool = nn.MaxPool3d(kernel_size=pooling_stride, stride=pooling_stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class UpLayer3D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        in_channels_low: Optional[int] = None,
        num_convs: int = 2,
        upsampling_stride: int = 2,
        # TODO: implement trilinear if needed
    ):
        super(UpLayer3D, self).__init__()

        if in_channels_low is None:
            in_channels_low = out_channels

        self.up = nn.ConvTranspose3d(
            in_channels=in_channels, 
            out_channels=in_channels // 2, 
            kernel_size=upsampling_stride, 
            stride=upsampling_stride, 
        )
        self.conv = BasicConv3D(
            in_channels=in_channels // 2 + in_channels_low, 
            out_channels=out_channels, 
            num_convs=num_convs
        )

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        x_add = self.up(x_low)
        
        # Correct dimensions before concatenation
        diffX = x_high.shape[-3] - x_add.shape[-3]
        diffY = x_high.shape[-2] - x_add.shape[-2]
        diffZ = x_high.shape[-1] - x_add.shape[-1]

        x_add = F.pad(x_add, (diffX // 2, diffX - diffX // 2, 
                              diffY // 2, diffY - diffY // 2, 
                              diffZ // 2, diffZ - diffZ // 2))

        return self.conv(torch.cat([x_high, x_add], dim=-4))


class UNetHiddenOnly3D(nn.Module):
    def __init__(
        self, 
        top_channels: int,
        num_lower_levels: int = 3,
        channels_per_lower_level: List[int] = [64, 96, 128],
        num_convs_per_level: int = 2,
        pooling_strides: Union[int, List[int]] = 2,
    ):
        super(UNetHiddenOnly3D, self).__init__()
        self.num_lower_levels = num_lower_levels
        self.channels_per_level = [top_channels] + channels_per_lower_level
        self.num_convs_per_level = num_convs_per_level
        if isinstance(pooling_strides, int):
            self.pooling_strides = [pooling_strides] * self.num_lower_levels
        else:
            self.pooling_strides = pooling_strides

        assert len(self.pooling_strides) == self.num_lower_levels, "The number of levels and strides must be the same"
        assert all(stride > 0 for stride in self.pooling_strides), "The strides must all be positive"

        self._build_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_buffer = [x]
        for enc in self.encoder_layers:
            x_buffer.append(enc(x_buffer[-1]))
        for dec in reversed(self.decoder_layers):
            x = dec(x_buffer.pop(), x_buffer[-1])
        return x

    def _build_layers(self):
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(self.num_lower_levels):
            self.encoder_layers.append(DownLayer3D(
                in_channels=self.channels_per_level[i], 
                out_channels=self.channels_per_level[i+1], 
                num_convs=self.num_convs_per_level, 
                pooling_stride=self.pooling_strides[i]
            ))
            self.decoder_layers.append(UpLayer3D(
                in_channels=self.channels_per_level[i+1], 
                out_channels=self.channels_per_level[i], 
                num_convs=self.num_convs_per_level, 
                upsampling_stride=self.pooling_strides[i]
            ))

if __name__ == "__main__":
    x = torch.randn(1, 1, 100, 100, 100)
    model = UNetHiddenOnly3D(top_channels=1, num_lower_levels=3, channels_per_lower_level=[64, 96, 128], num_convs_per_level=2, pooling_strides=2)
    print(model(x).shape)
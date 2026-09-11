import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv -> Norm -> ReLU block used by the UNet-style decoder."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SeparableConvBNAct(nn.Module):
    """Depthwise separable Conv -> Norm -> ReLU block.

    This is a drop-in lightweight replacement for a standard 3x3 ConvBNReLU.
    A 1x1 pointwise convolution mixes channels after the depthwise spatial
    convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size,
                      padding=padding, groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """Two lightweight refinement blocks, following the common UNet design."""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,
                 use_depthwise=True):
        super().__init__()
        block = SeparableConvBNAct if use_depthwise else ConvBNAct
        self.block = nn.Sequential(
            block(in_channels, out_channels, 3, norm_layer),
            block(out_channels, out_channels, 3, norm_layer),
        )

    def forward(self, x):
        return self.block(x)


class UpFuseBlock(nn.Module):
    """
    One UNet decoding stage:
        1. upsample the low-resolution feature to the skip feature size;
        2. project the skip feature to the target channel dimension;
        3. concatenate and refine with DoubleConv.
    """

    def __init__(self, in_channels, skip_channels, out_channels,
                 norm_layer=nn.BatchNorm2d, align_corners=False,
                 use_depthwise=True):
        super().__init__()
        self.align_corners = align_corners

        self.skip_proj = ConvBNAct(skip_channels, out_channels, 1, norm_layer)
        self.fuse = DoubleConv(in_channels + out_channels, out_channels,
                               norm_layer=norm_layer,
                               use_depthwise=use_depthwise)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        skip = self.skip_proj(skip)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class UNetDecoder(nn.Module):
    """
    UNet-style decoder for four multi-scale encoder features.

    Default input order is the same as the current MiT encoder:
        inputs = [c1, c2, c3, c4]
        c1: highest resolution, normally 1/4 of the original image
        c2: 1/8
        c3: 1/16
        c4: lowest resolution, normally 1/32

    If your feature list is already ordered from low resolution to high
    resolution, instantiate this module with input_order='low_to_high' and pass
    in_channels in the same order as the input feature list.

    The decoder itself fuses features from low resolution to high resolution:
        c4 -> c3 -> c2 -> c1

    Output:
        logits at c1 resolution, i.e. normally 1/4 of the original image.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 decoder_channels=[192, 96, 48],
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 align_corners=False,
                 input_order='high_to_low',
                 use_depthwise=True):
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(
                f"UNetDecoder expects 4 input channel values, got {len(in_channels)}."
            )
        if len(decoder_channels) != 3:
            raise ValueError(
                "decoder_channels should contain 3 values, one for each "
                "low-to-high fusion stage: c4->c3, c3->c2, c2->c1."
            )
        if input_order not in ('high_to_low', 'low_to_high'):
            raise ValueError(
                "input_order must be either 'high_to_low' or 'low_to_high'."
            )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        self.align_corners = align_corners
        self.input_order = input_order
        self.use_depthwise = use_depthwise

        if input_order == 'high_to_low':
            c1_channels, c2_channels, c3_channels, c4_channels = in_channels
        else:
            c4_channels, c3_channels, c2_channels, c1_channels = in_channels
        d3_channels, d2_channels, d1_channels = decoder_channels

        # Start from the lowest-resolution feature c4.
        self.bottleneck = DoubleConv(
            c4_channels,
            d3_channels,
            norm_layer=norm_layer,
            use_depthwise=use_depthwise,
        )

        # Low -> high progressive fusion.
        self.up3 = UpFuseBlock(
            in_channels=d3_channels,
            skip_channels=c3_channels,
            out_channels=d3_channels,
            norm_layer=norm_layer,
            align_corners=align_corners,
            use_depthwise=use_depthwise,
        )
        self.up2 = UpFuseBlock(
            in_channels=d3_channels,
            skip_channels=c2_channels,
            out_channels=d2_channels,
            norm_layer=norm_layer,
            align_corners=align_corners,
            use_depthwise=use_depthwise,
        )
        self.up1 = UpFuseBlock(
            in_channels=d2_channels,
            skip_channels=c1_channels,
            out_channels=d1_channels,
            norm_layer=norm_layer,
            align_corners=align_corners,
            use_depthwise=use_depthwise,
        )

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(d1_channels, num_classes, kernel_size=1)

        self._init_weights()

    def _split_inputs(self, inputs):
        if len(inputs) != 4:
            raise ValueError(f"UNetDecoder expects 4 feature maps, got {len(inputs)}.")

        if self.input_order == 'high_to_low':
            c1, c2, c3, c4 = inputs
        else:
            c4, c3, c2, c1 = inputs

        return c1, c2, c3, c4

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        c1, c2, c3, c4 = self._split_inputs(inputs)

        x = self.bottleneck(c4)  # 1/32
        x = self.up3(x, c3)      # 1/16
        x = self.up2(x, c2)      # 1/8
        x = self.up1(x, c1)      # 1/4

        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# Alias with a "Head" suffix, consistent with other decoder files.
UNetHead = UNetDecoder


if __name__ == '__main__':
    batch_size = 2
    in_channels = [64, 128, 320, 512]
    features = [
        torch.randn(batch_size, 64, 64, 128),   # c1, 1/4
        torch.randn(batch_size, 128, 32, 64),   # c2, 1/8
        torch.randn(batch_size, 320, 16, 32),   # c3, 1/16
        torch.randn(batch_size, 512, 8, 16),    # c4, 1/32
    ]

    model = UNetDecoder(in_channels=in_channels, num_classes=12)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    with torch.no_grad():
        output = model(features)
        print("输出形状:", output.shape)  # [2, 12, 64, 128], i.e. 1/4 scale

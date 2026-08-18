"""NAFNet (Simple Baselines for Image Restoration, ECCV 2022) adapted for 2x SR.

Body runs at LR resolution (plain NAFBlock stack, no U-Net downsampling since
inputs are only 128x128), followed by a PixelShuffle 2x head. Global residual:
bicubic-upsampled input is added to the output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.conv2(self.conv1(self.norm1(x)))
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.sg(self.conv4(self.norm2(x)))
        y = self.conv5(y)
        return x + y * self.gamma


class NAFNetSR(nn.Module):
    def __init__(self, width=64, num_blocks=32, upscale=2):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.up = nn.Sequential(
            nn.Conv2d(width, width * upscale * upscale, 3, padding=1),
            nn.PixelShuffle(upscale),
        )
        self.out = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)
        f = self.intro(x)
        f = self.body(f) + f
        f = self.up(f)
        return self.out(f) + base

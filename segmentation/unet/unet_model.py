#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from .unet_func import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # 572×572×1 conv=> 570×570×64 conv=> 568×568×64
        self.inc = inconv(n_channels, 64)

        # 568×568×64 pool=> 284×284×128 conv=> 282×282×128 conv=> 280×280×128
        self.down1 = down(64, 128)

        # 280×280×128 pool=> 140×140×128 conv=> 138×138×256 conv=> 136×136×256
        self.down2 = down(128, 256)

        # 136×136×256 pool=> 68×68×256 conv=> 66×66×512 conv=> 64×64×512
        self.down3 = down(256, 512)

        # 64×64×512 pool=> 32×32×512 conv=> 30×30×1024 conv=> 28×28×1024
        self.down4 = down(512, 1024)

        # 28×28×1024 upconv=> 56×56×512 concat=> 56×56×1024 conv=> 54×54×512 conv=> 52×52×512
        self.up1 = up(1024, 512)

        # 52×52×512 upconv=> 104×104×256 concat=> 104×104×512 conv=> 102×102×256 conv=> 100×100×256
        self.up2 = up(512, 256)

        # 100×100×256 upconv=> 200×200×128 concat=> 200×200×256 conv=> 198×198×128 conv=> 196×196×128
        self.up3 = up(256, 128)

        # 196×196×128 upconv=> 392×392×64 concat=> 392×392×128 conv=> 390×390×64 conv=> #388×388×64 => 388×388×1
        self.up4 = up(128, 64)

        # 388×388×64 => 388×388×1
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

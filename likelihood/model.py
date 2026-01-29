import torch
import torch.nn as nn

import MinkowskiEngine as ME

KS = (1, 3, 3)
DS = (1, 2, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3):
        super().__init__()
        self.c1 = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=KS, dimension=dim)
        self.b1 = ME.MinkowskiBatchNorm(out_ch)
        self.c2 = ME.MinkowskiConvolution(out_ch, out_ch, kernel_size=KS, dimension=dim)
        self.b2 = ME.MinkowskiBatchNorm(out_ch)
        self.r = ME.MinkowskiReLU(inplace=True)
        self.sc = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=1, dimension=dim) if in_ch != out_ch else None

    def forward(self, x):
        i = x if self.sc is None else self.sc(x)
        x = self.r(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return self.r(x + i)


class InputNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(c))
        self.log_scale = nn.Parameter(torch.zeros(c))

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        F = (x.F - self.shift) * self.log_scale.exp()
        return ME.SparseTensor(features=F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)


class MinkUNetClassifier(nn.Module):
    def __init__(self, in_channels=4, base=32, strides=4, dropout=0.2):
        super().__init__()
        self.inorm = InputNorm(in_channels)
        self.c0 = ME.MinkowskiConvolution(in_channels, base, kernel_size=KS, dimension=3)

        self.enc = nn.ModuleList()
        ch = base
        for _ in range(strides):
            self.enc.append(ResidualBlock(ch, ch * 2))
            self.enc.append(ME.MinkowskiConvolution(ch * 2, ch * 2, kernel_size=DS, stride=DS, dimension=3))
            ch *= 2

        self.mid = ResidualBlock(ch, ch)

        self.dec = nn.ModuleList()
        for i in range(strides):
            up = ch // 2
            self.dec.append(ME.MinkowskiConvolutionTranspose(ch, up, kernel_size=DS, stride=DS, dimension=3))
            skip = base * (2 ** (strides - i))
            self.dec.append(ResidualBlock(up + skip, up))
            ch = up

        self.bn = nn.Sequential(ME.MinkowskiBatchNorm(base), ME.MinkowskiReLU(inplace=True))
        self.drop = ME.MinkowskiDropout(dropout)
        self.p_sum = ME.MinkowskiGlobalPooling()
        self.p_max = ME.MinkowskiGlobalMaxPooling()

        h = base * 2
        self.head = nn.Sequential(
            nn.Linear(h + 1, h),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:
        bs = int(x.C[:, 0].max().item()) + 1
        cnt = torch.bincount(x.C[:, 0], minlength=bs).to(dtype=torch.float32)
        log_cnt = torch.log1p(cnt).view(bs, 1)

        x = self.inorm(x)
        x = self.c0(x)
        skips = []
        for i in range(0, len(self.enc), 2):
            x = self.enc[i](x)
            skips.append(x)
            x = self.enc[i + 1](x)

        x = self.mid(x)

        for i in range(0, len(self.dec), 2):
            x = self.dec[i](x)
            x = ME.cat(x, skips.pop())
            x = self.dec[i + 1](x)

        x = self.drop(self.bn(x))
        s = self.p_sum(x).F
        m = self.p_max(x).F
        z = torch.cat([s, m], dim=1)
        z = torch.cat([z, log_cnt.to(z.device, non_blocking=True)], dim=1)
        return self.head(z)

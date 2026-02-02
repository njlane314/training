import torch.nn as nn

import MinkowskiEngine as ME


class ResBlock(nn.Module):
    def __init__(self, cin, cout, D=3, ks=(3, 3, 3)):
        super().__init__()
        self.conv1 = ME.MinkowskiSubmanifoldConvolution(
            cin, cout, kernel_size=ks, dimension=D, bias=False
        )
        self.bn1 = ME.MinkowskiBatchNorm(cout)
        self.conv2 = ME.MinkowskiSubmanifoldConvolution(
            cout, cout, kernel_size=ks, dimension=D, bias=False
        )
        self.bn2 = ME.MinkowskiBatchNorm(cout)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.proj = None
        if cin != cout:
            self.proj = nn.Sequential(
                ME.MinkowskiLinear(cin, cout, bias=False),
                ME.MinkowskiBatchNorm(cout),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class SparseUResNetEncoderClassifier(nn.Module):
    """
    UResNet-style residual encoder + global pooling head for event-level LLR.
    Input coords: (batch, plane, y, x) with D=3 spatial dims (plane,y,x).
    """

    def __init__(self, in_ch=3, base=32, D=3):
        super().__init__()
        self.D = D

        # Keep it simple: isotropic kernels, downsample only in (y,x)
        self.stem = nn.Sequential(
            ME.MinkowskiSubmanifoldConvolution(
                in_ch, base, kernel_size=3, dimension=D, bias=False
            ),
            ME.MinkowskiBatchNorm(base),
            ME.MinkowskiReLU(inplace=True),
        )

        self.b0 = ResBlock(base, base, D=D)
        self.down1 = ME.MinkowskiConvolution(
            base,
            base * 2,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            dimension=D,
            bias=False,
        )
        self.b1 = ResBlock(base * 2, base * 2, D=D)

        self.down2 = ME.MinkowskiConvolution(
            base * 2,
            base * 4,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            dimension=D,
            bias=False,
        )
        self.b2 = ResBlock(base * 4, base * 4, D=D)

        self.down3 = ME.MinkowskiConvolution(
            base * 4,
            base * 8,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            dimension=D,
            bias=False,
        )
        self.b3 = ResBlock(base * 8, base * 8, D=D)

        self.pool = ME.MinkowskiGlobalMaxPooling()
        self.head = nn.Linear(base * 8, 1)

    def forward(self, x: ME.SparseTensor):
        x = self.stem(x)
        x = self.b0(x)
        x = self.b1(self.down1(x))
        x = self.b2(self.down2(x))
        x = self.b3(self.down3(x))
        x = self.pool(x)  # one feature vector per batch item
        return self.head(x.F).squeeze(1)  # logits

import torch.nn as nn

import MinkowskiEngine as ME

class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = ME.MinkowskiSubmanifoldConvolution(
            cin, cout, kernel_size=(3, 3), dimension=2, bias=False
        )
        self.bn1 = ME.MinkowskiBatchNorm(cout)
        self.conv2 = ME.MinkowskiSubmanifoldConvolution(
            cout, cout, kernel_size=(3, 3), dimension=2, bias=False
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
    2D residual encoder + global max pooling head for per-plane logits.
    Input coords: (batch, y, x) with D=2 spatial dims.
    """

    def __init__(self, in_ch=2, base=32):
        super().__init__()

        self.stem = nn.Sequential(
            ME.MinkowskiSubmanifoldConvolution(
                in_ch, base, kernel_size=(3, 3), dimension=2, bias=False
            ),
            ME.MinkowskiBatchNorm(base),
            ME.MinkowskiReLU(inplace=True),
        )

        self.b0 = ResBlock(base, base)
        self.down1 = ME.MinkowskiConvolution(
            base,
            base * 2,
            kernel_size=(2, 2),
            stride=(2, 2),
            dimension=2,
            bias=False,
        )
        self.b1 = ResBlock(base * 2, base * 2)

        self.down2 = ME.MinkowskiConvolution(
            base * 2,
            base * 4,
            kernel_size=(2, 2),
            stride=(2, 2),
            dimension=2,
            bias=False,
        )
        self.b2 = ResBlock(base * 4, base * 4)

        self.down3 = ME.MinkowskiConvolution(
            base * 4,
            base * 8,
            kernel_size=(2, 2),
            stride=(2, 2),
            dimension=2,
            bias=False,
        )
        self.b3 = ResBlock(base * 8, base * 8)

        self.pool = ME.MinkowskiGlobalMaxPooling()
        self.head = nn.Linear(base * 8, 1)

    def forward(self, x: ME.SparseTensor):
        x = self.stem(x)
        x = self.b0(x)
        x = self.b1(self.down1(x))
        x = self.b2(self.down2(x))
        x = self.b3(self.down3(x))
        x = self.pool(x)  # one feature vector per batch item
        return self.head(x.F)  # logits [B, 1]

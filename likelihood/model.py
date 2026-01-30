import torch
import torch.nn as nn

import MinkowskiEngine as ME

# Allow cross-view convolution (mix U/V/W through the "view" coordinate axis)
KS = (3, 3, 3)
KS_VIEW_MIX = (3, 1, 1)
DS = (1, 2, 2)


class MinkowskiLayerNorm(nn.Module):
    """
    @brief LayerNorm over channels for MinkowskiEngine SparseTensors.

    This avoids BatchNorm running-stat instability on sparse, highly variable nnz events.
    """

    def __init__(self, c: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps, elementwise_affine=affine)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        F = self.ln(x.F)
        return ME.SparseTensor(
            features=F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )


class ResidualBlock(nn.Module):
    """
    @brief Residual block for sparse convolutional features with an optional skip projection.
    """

    def __init__(self, in_ch, out_ch, dim=3):
        """
        @brief Initialise the residual block layers and optional shortcut convolution.
        """
        super().__init__()
        self.c1 = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=KS, dimension=dim)
        self.n1 = MinkowskiLayerNorm(out_ch)
        self.c2 = ME.MinkowskiConvolution(out_ch, out_ch, kernel_size=KS, dimension=dim)
        self.n2 = MinkowskiLayerNorm(out_ch)
        self.r = ME.MinkowskiReLU(inplace=True)
        self.sc = ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=1, dimension=dim) if in_ch != out_ch else None

    def forward(self, x):
        """
        @brief Apply the residual block and return the activated sum.
        """
        i = x if self.sc is None else self.sc(x)
        x = self.r(self.n1(self.c1(x)))
        x = self.n2(self.c2(x))
        return self.r(x + i)


class InputNorm(nn.Module):
    """
    @brief Learnable input normalisation for sparse tensors.
    """

    def __init__(self, c):
        """
        @brief Initialise per-channel shift and scale parameters.
        """
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(c))
        self.log_scale = nn.Parameter(torch.zeros(c))

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        @brief Apply normalisation to sparse tensor features.
        """
        F = (x.F - self.shift) * self.log_scale.exp()
        return ME.SparseTensor(features=F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)


class MinkUNetClassifier(nn.Module):
    """
    @brief Sparse UNet classifier with global pooling and a compact classification head.
    """

    def __init__(self, in_channels=4, base=32, strides=4, dropout=0.2):
        """
        @brief Build the encoder-decoder backbone and classification head.
        """
        super().__init__()
        self.inorm = InputNorm(in_channels)
        self.c0 = ME.MinkowskiConvolution(in_channels, base, kernel_size=KS, dimension=3)
        self.view_mix = ME.MinkowskiConvolution(base, base, kernel_size=KS_VIEW_MIX, dimension=3)
        self.view_mix.kernel_size = KS_VIEW_MIX

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

        self.bn = nn.Sequential(MinkowskiLayerNorm(base), ME.MinkowskiReLU(inplace=True))
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
        """
        @brief Run the forward pass, pool globally, and emit logits.
        """
        batch_ids = x.C[:, 0].to(dtype=torch.int64)
        uniq, cnt = torch.unique(batch_ids, sorted=True, return_counts=True)

        x = self.inorm(x)
        x = self.c0(x)
        x = self.view_mix(x)
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
        if torch.all(cnt == 1):
            p_sum = x
            p_max = x
        else:
            p_sum = self.p_sum(x)
            p_max = self.p_max(x)
        s = p_sum.F
        m = p_max.F
        out_batch_ids = p_sum.C[:, 0].to(dtype=torch.int64)
        if out_batch_ids.numel() > 1:
            order = torch.argsort(out_batch_ids)
            s = s[order]
            m = m[order]
            out_batch_ids = out_batch_ids[order]
        z = torch.cat([s, m], dim=1)
        pos = torch.searchsorted(uniq, out_batch_ids)
        log_cnt = torch.log1p(cnt[pos].to(dtype=torch.float32))
        z = torch.cat([z, log_cnt.to(z.device, non_blocking=True).view(-1, 1)], dim=1)
        return self.head(z)

"""
1D U-Net resonance detector for full VNA sweeps.

Outputs per sample: a heatmap logit (resonance center probability), a
sub-sample offset, and log10(Qr). Fully convolutional, so any input length
that is a multiple of 2**depth works. The architecture must match the
released weights (``vna.cnn.model``); the network is trained
separately.
"""
import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Two 1D convolutions with group norm and GELU.

    Attributes:
    net (nn.Sequential): the layers.
    """
    def __init__(self, c_in, c_out, kernel = 7, dilation = 1):
        """
        Build the block.

        Parameters:
        c_in (int): input channels.
        c_out (int): output channels.
        kernel (int): kernel size (odd). Default 7.
        dilation (int): dilation of the second convolution. Default 1.
        """
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding = pad),
            nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel, padding = pad * dilation,
                      dilation = dilation),
            nn.GroupNorm(min(8, c_out), c_out), nn.GELU())

    def forward(self, x):
        """
        Apply the block.

        Parameters:
        x (torch.Tensor): shape (B, c_in, L).

        Returns:
        y (torch.Tensor): shape (B, c_out, L).
        """
        return self.net(x)

class UNet1D(nn.Module):
    """
    1D U-Net with heatmap, offset and log10(Qr) heads.

    Attributes:
    depth (int): number of downsampling levels (input length must be a
        multiple of 2**depth).
    """
    def __init__(self, c_in = 3, width = 24, depth = 7, max_width = 192,
                 kernel = 7):
        """
        Build the network.

        Parameters:
        c_in (int): input channels. Default 3 (``infer.detector_features``).
        width (int): channels at full resolution. Default 24.
        depth (int): downsampling levels. Default 7.
        max_width (int): channel cap. Default 192.
        kernel (int): convolution kernel size. Default 7.
        """
        super().__init__()
        self.depth = depth
        ch = [min(width * 2 ** i, max_width) for i in range(depth + 1)]
        self.inp = ConvBlock(c_in, ch[0], kernel)
        self.down = nn.ModuleList(ConvBlock(ch[i], ch[i + 1], kernel,
                                            dilation = 2)
                                  for i in range(depth))
        self.up = nn.ModuleList(ConvBlock(ch[i + 1] + ch[i], ch[i], kernel)
                                for i in reversed(range(depth)))
        self.head = nn.Conv1d(ch[0], 3, 1)

    def forward(self, x):
        """
        Run the network.

        Parameters:
        x (torch.Tensor): shape (B, c_in, L), L a multiple of 2**depth.

        Returns:
        heat_logit (torch.Tensor): shape (B, L).
        offset (torch.Tensor): shape (B, L), sub-sample offset in samples.
        log_q (torch.Tensor): shape (B, L), log10(Qr).
        """
        skips = []
        h = self.inp(x)
        for blk in self.down:
            skips.append(h)
            h = blk(F.avg_pool1d(h, 2))
        for blk in self.up:
            s = skips.pop()
            h = F.interpolate(h, size = s.shape[-1], mode = 'linear',
                              align_corners = False)
            h = blk(torch.cat((h, s), dim = 1))
        out = self.head(h)
        return out[:, 0], out[:, 1], out[:, 2] + 4.

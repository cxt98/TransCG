"""
Depth Filler Network.

Author: Hongjie Fang.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .dense import DenseBlock
from .duc import DenseUpsamplingConvolution
import torchvision

class DFNet(nn.Module):
    """
    Depth Filler Network (DFNet).
    """
    def __init__(self, in_channels = 4, hidden_channels = 64, L = 5, k = 12, use_DUC = True, **kwargs):
        super(DFNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.L = L
        self.k = k
        self.use_DUC = use_DUC
        # First
        self.first = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True)
        )
        # Dense1: skip
        self.CDC1 = CDCX(self.hidden_channels, self.hidden_channels)
        # Dense1: normal
        self.CDCD1 = CDCX(self.hidden_channels, self.hidden_channels, stride=2)
        # Dense2: skip
        self.CDC2 = CDCX(self.hidden_channels, self.hidden_channels)
        # Dense2: normal
        self.CDCD2 = CDCX(self.hidden_channels, self.hidden_channels, stride=2)
        # Dense3: skip
        self.CDC3 = CDCX(self.hidden_channels, self.hidden_channels)
        # Dense3: normal
        self.CDCD3 = CDCX(self.hidden_channels, self.hidden_channels, stride=2)
        # Dense4
        self.CDC4 = CDCX(self.hidden_channels, self.hidden_channels)
        # DUC upsample 1
        self.CDCU1 = CDCX(self.hidden_channels, self.hidden_channels, upscale_factor=2)
        # DUC upsample 2
        self.CDCU2 = CDCX(self.hidden_channels * 2, self.hidden_channels, upscale_factor=2)
        # DUC upsample 3
        self.CDCU3 = CDCX(self.hidden_channels * 2, self.hidden_channels, upscale_factor=2)
        # DUC upsample 4
        self.CDCU4 = CDCX(self.hidden_channels * 2, self.hidden_channels, upscale_factor=2)
        # Final
        self.final = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(True)
        )
        # also output a confidence/weight/error map
        self.final_wmap = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, 1, kernel_size = 3, stride = 1, padding = 1)
        )
    
    def _make_upconv(self, in_channels, out_channels, upscale_factor = 2):
        if self.use_DUC:
            return DenseUpsamplingConvolution(in_channels, out_channels, upscale_factor = upscale_factor)
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size = upscale_factor, stride = upscale_factor, padding = 0, output_padding = 0),
                nn.BatchNorm2d(out_channels, out_channels),
                nn.ReLU(True)
            )
    
    def forward(self, rgb, depth):
        n, H, W = depth.shape
        depth = depth.view(n, 1, H, W)
        h = self.first(torch.cat((rgb, depth), dim = 1))
        # tested, torch.nn.functional.interpolate() doesn't support nearest-exact mode, change to torchvision API and verified to work the same as PIL
        # depth1 = F.interpolate(depth, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        depth1 = torchvision.transforms.Resize((H//2, W//2), torchvision.transforms.InterpolationMode.NEAREST)(depth)
        # dense1: skip
        h_d1s = self.CDC1(h, depth1)
        # dense1: normal
        h = self.CDCD1(h, depth1)
        # h = self.dense1_conv1(h)
        # depth2 = F.interpolate(depth1, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        depth2 = torchvision.transforms.Resize((H//4, W//4), torchvision.transforms.InterpolationMode.NEAREST)(depth1)
        # dense2: skip
        h_d2s = self.CDC2(h, depth2)
        # dense2: normal
        h = self.CDCD2(h, depth2)
        # depth3 = F.interpolate(depth2, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        depth3 = torchvision.transforms.Resize((H//8, W//8), torchvision.transforms.InterpolationMode.NEAREST)(depth2)
        # dense3: skip
        h_d3s = self.CDC3(h, depth3)
        # dense3: normal
        h = self.CDCD3(h, depth3)
        # depth4 = F.interpolate(depth3, scale_factor = 0.5, mode = "bilinear", align_corners = True)
        depth4 = torchvision.transforms.Resize((H//16, W//16), torchvision.transforms.InterpolationMode.NEAREST)(depth3)
        h = self.CDC4(h, depth4)
        h = self.CDCU1(h, depth4)
        h = torch.cat((h, h_d3s), dim = 1)
        h = self.CDCU2(h, depth3)
        h = torch.cat((h, h_d2s), dim = 1)
        h = self.CDCU3(h, depth2)
        h = torch.cat((h, h_d1s), dim = 1)
        h = self.CDCU4(h, depth1)
        # final
        d = self.final(h).squeeze(1)
        w = self.final_wmap(h).squeeze(1)

        return d, w # depth prediction, error map

class CDCX(nn.Module):
    # (default) CDC: stride=1, CDCD: stride=2, CDCU: upscale_factor=2
    def __init__(self, in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, L=5, k=12, upscale_factor=1):
        super(CDCX, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, 1, padding),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True)
        )
        self.dense = DenseBlock(hidden_channels, L, k, with_bn=True)
        if upscale_factor == 1: # CDC or CDCD
            self.conv2 = nn.Sequential(
                nn.Conv2d(k, hidden_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(True)
            )
        elif upscale_factor == 2: # CDCU
            self.conv2 = nn.Sequential(
                nn.Conv2d(k, hidden_channels * upscale_factor * upscale_factor, kernel_size, 1, padding),
                nn.BatchNorm2d(hidden_channels * upscale_factor * upscale_factor),
                nn.ReLU(True),
                nn.PixelShuffle(upscale_factor)
            )
    
    def forward(self, feature, depth):
        feature = self.conv1(feature)
        feature = self.dense(torch.cat((feature, depth), dim=1))
        feature = self.conv2(feature)
        return feature
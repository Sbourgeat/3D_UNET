"""
3D UNET implemented by Eva Frossard, EPFL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import the functional module


class UNet3D(nn.Module):
    def __init__(self, dropout):
        super(UNet3D, self).__init__()
        self.dropout = dropout

        # Encoder
        self.encoder1 = self.contract_block(1, 32, self.dropout)
        self.encoder2 = self.contract_block(32, 64, self.dropout)
        self.encoder3 = self.contract_block(64, 128, self.dropout)

        # Decoder
        self.decoder3 = self.expand_block(128, 64, self.dropout)
        self.decoder2 = self.expand_block(128, 32, self.dropout)
        self.decoder1 = self.expand_block(64, 32, self.dropout)

        # Output layer
        self.output_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid to have prediction values between 0 and 1

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # Decoder
        dec3 = self.decoder3(enc3)
        # Example of using F.pad for dimension matching
        if dec3.size(2) != enc2.size(2):
            target_size = enc2.size(2)
            dec3 = F.interpolate(dec3, size=(target_size, target_size, target_size))

        dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
        dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))

        # Output layer
        out = self.output_conv(dec1)
        out = self.sigmoid(out)  # Sigmoid to have prediction values between 0 and 1
        out = out.permute(0, 2, 3, 4, 1)

        return out

    def contract_block(self, in_channels, out_channels, dropout):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        return block

    def expand_block(self, in_channels, out_channels, dropout):
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # Adjusting dimensions for 3D convolutions

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # Decoder
        dec3 = self.decoder3(enc3)
        dec3 = self.crop_and_concat(enc2, dec3)

        dec2 = self.decoder2(dec3)
        dec2 = self.crop_and_concat(enc1, dec2)

        dec1 = self.decoder1(dec2)

        # Output layer
        out = self.output_conv(dec1)
        return out  # Logits are returned directly

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

    def crop_and_concat(self, upsampled, bypass):
        # This function crops the 'bypass' tensor to the size of 'upsampled' tensor
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = bypass[
            :,
            :,
            c : bypass.size()[2] - c,
            c : bypass.size()[3] - c,
            c : bypass.size()[4] - c,
        ]
        return torch.cat((upsampled, bypass), 1)

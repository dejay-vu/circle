import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from vector_quantize_pytorch import VectorQuantize


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.net(x))


class Encoder(nn.Module):
    """
    (B, 3, 84, 84) → (B, 16, 512)

    - 4 conv layers
    - 2 residual blocks per layer
    - 21x21 feature map partitioned into 4x4 non-overlapping 5x5 windows -> 16 tokens
    """

    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.patch = 5

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(64),
            ResidualBlock(64),
        )  # 84 → 42

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )  # 42 → 21

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            ResidualBlock(256),
            ResidualBlock(256),
        )  # 21 → 21

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )  # 21 → 21

        self.projection = nn.Linear(
            self.patch * self.patch * out_channels, out_channels
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # (B, 512, 21, 21)
        x = x[:, :, 0:20, 0:20]
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch,
            p2=self.patch,
        )  # (B, 16, 5*5*512)

        x = self.projection(x)  # (B, 16, 512)

        return x  # (B, 16, 512)


class Decoder(nn.Module):
    """
    (B, 16, 512) → (B, 3, 84, 84)

    - reverse of Encoder
    - 16 tokens -> 4x4 grid of 5x5 patches -> (B, 512, 21, 21)
    - 4 deconv layers
    """

    def __init__(self, in_channels=512, out_channels=3):
        super().__init__()
        self.patch = 5

        self.projection = nn.Linear(
            in_channels, self.patch * self.patch * in_channels
        )

        self.deconv4 = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
        )  # 21 → 21

        self.deconv3 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.SiLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        )  # 21 → 21

        self.deconv2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        )  # 21 → 42

        self.deconv1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.SiLU(),
            nn.ConvTranspose2d(
                64, out_channels, kernel_size=4, stride=2, padding=1
            ),
        )  # 42 → 84

    def forward(self, x):
        # x: (B, 16, 512)
        x = self.projection(x)  # (B, 16, 5×5×512)
        x = x.view(
            x.size(0), 4, 4, 512, self.patch, self.patch
        )  # (B, 4, 4, 512, 5, 5)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, 512, 4, 5, 4, 5)
        x = x.reshape(x.size(0), 512, 4 * 5, 4 * 5)  # (B, 512, 20, 20)

        # 填充到 21×21（补 1 行 1 列）
        x = F.pad(x, (0, 1, 0, 1))  # (B, 512, 21, 21)

        x = self.deconv4(x)  # (B, 256, 21, 21)
        x = self.deconv3(x)  # (B, 128, 21, 21)
        x = self.deconv2(x)  # (B, 64, 42, 42)
        x = self.deconv1(x)  # (B, 3, 84, 84)

        return x


vq_vae = VectorQuantize(
    dim=512,
    codebook_size=512,  # each table smaller
    decay=0.8,
    commitment_weight=0.1,
)

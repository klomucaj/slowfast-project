import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    """
    A basic 3D convolutional block: Conv3D -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SlowPathway(nn.Module):
    """
    Slow pathway: Processes low-frame-rate inputs with high channel depth.
    """
    def __init__(self, input_channels=3, base_channels=64):
        super(SlowPathway, self).__init__()
        self.layer1 = BasicBlock3D(input_channels, base_channels, stride=(1, 2, 2))
        self.layer2 = BasicBlock3D(base_channels, base_channels * 2, stride=(1, 2, 2))
        self.layer3 = BasicBlock3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2))

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

class FastPathway(nn.Module):
    """
    Fast pathway: Processes high-frame-rate inputs with low channel depth.
    """
    def __init__(self, input_channels=3, base_channels=8):
        super(FastPathway, self).__init__()
        self.layer1 = BasicBlock3D(input_channels, base_channels, stride=(1, 2, 2))
        self.layer2 = BasicBlock3D(base_channels, base_channels * 2, stride=(1, 2, 2))
        self.layer3 = BasicBlock3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2))

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))

if __name__ == "__main__":
    # Testing SlowPathway and FastPathway
    slow_path = SlowPathway()
    fast_path = FastPathway()

    x_slow = torch.randn(2, 3, 32, 224, 224)  # Slow pathway input
    x_fast = torch.randn(2, 3, 8, 224, 224)   # Fast pathway input

    print("Slow Pathway output shape:", slow_path(x_slow).shape)  # Should reduce spatial dimensions
    print("Fast Pathway output shape:", fast_path(x_fast).shape)

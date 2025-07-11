import torch
import torch.nn as nn
from datasets.models.backbone import SlowPathway, FastPathway
from datasets.models.lateral_connections import LateralConnections

class SlowFastModel(nn.Module):
    def __init__(self, num_classes=16, slow_channels=64, fast_channels=8, dropout_rate=0.3):
        """
        SlowFast Model integrating slow and fast pathways with lateral connections.

        Args:
            num_classes (int): Number of output classes.
            slow_channels (int): Base channel size for slow pathway.
            fast_channels (int): Base channel size for fast pathway.
            dropout_rate (float): Dropout rate before the final classifier.
        """
        super(SlowFastModel, self).__init__()

        # Slow pathway processes input at low frame rate (semantic info)
        self.slow_path = SlowPathway(input_channels=3, base_channels=slow_channels)

        # Fast pathway processes input at high frame rate (motion info)
        self.fast_path = FastPathway(input_channels=3, base_channels=fast_channels)

        # Fuse features from both pathways using lateral connections
        self.lateral_connections = LateralConnections(
            slow_channels=slow_channels * 4,
            fast_channels=fast_channels * 4
        )

        # Global average pooling to reduce spatiotemporal features to fixed size
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        # Dropout layer to regularize fully connected classifier
        self.dropout = nn.Dropout(p=dropout_rate)

        # Final fully connected classification layer
        # Input features: 512 (must match output channel size from lateral connections)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x_slow, x_fast):
        # Extract slow pathway features
        slow_features = self.slow_path(x_slow)

        # Extract fast pathway features
        fast_features = self.fast_path(x_fast)

        # Fuse features with lateral connections module
        combined_features = self.lateral_connections(slow_features, fast_features)

        # Global average pooling and flatten
        x = self.pool(combined_features)
        x = self.flatten(x)

        # Apply dropout before classifier
        x = self.dropout(x)

        # Classification logits
        logits = self.fc(x)
        return logits


if __name__ == "__main__":
    # Quick test to verify output shape
    model = SlowFastModel(num_classes=16, dropout_rate=0.3)
    x_slow = torch.randn(2, 3, 32, 224, 224)  # Slow pathway input [B, C, T, H, W]
    x_fast = torch.randn(2, 3, 8, 224, 224)   # Fast pathway input [B, C, T, H, W]
    output = model(x_slow, x_fast)
    print("Output shape:", output.shape)  # Should be [2, 16]

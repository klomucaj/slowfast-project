import torch
import torch.nn as nn

class LateralConnections(nn.Module):
    """
    Connects Slow and Fast pathways using lateral feature fusion.
    """
    def __init__(self, slow_channels, fast_channels):
        super(LateralConnections, self).__init__()
        # Reduce fast pathway channels to match slow pathway dimensions
        self.fast_to_slow = nn.Conv3d(fast_channels, slow_channels, kernel_size=1)

    def forward(self, slow_features, fast_features):
        """
        Args:
            slow_features: Output from SlowPathway.
            fast_features: Output from FastPathway.
        Returns:
            Combined features.
        """
        # Upsample fast pathway features to match slow spatial resolution
        fast_resized = nn.functional.interpolate(
            self.fast_to_slow(fast_features), 
            size=slow_features.shape[2:], 
            mode="trilinear", 
            align_corners=False
        )

        # Combine slow and resized fast features
        combined_features = torch.cat([slow_features, fast_resized], dim=1)
        return combined_features

if __name__ == "__main__":
    # Testing Lateral Connections
    lateral = LateralConnections(slow_channels=256, fast_channels=32)

    slow_features = torch.randn(2, 256, 8, 28, 28)  # Slow pathway features
    fast_features = torch.randn(2, 32, 16, 14, 14)  # Fast pathway features (higher temporal resolution)

    combined = lateral(slow_features, fast_features)
    print("Combined features shape:", combined.shape)  # Should match [2, slow_channels + reduced fast_channels, ...]

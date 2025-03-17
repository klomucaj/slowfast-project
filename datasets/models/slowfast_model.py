import torch
import torch.nn as nn
from datasets.models.backbone import SlowPathway, FastPathway
from datasets.models.lateral_connections import LateralConnections

class SlowFastModel(nn.Module):
    """
    SlowFast model architecture for video classification.
    """
    def __init__(self, num_classes=10, slow_channels=64, fast_channels=8):
        super(SlowFastModel, self).__init__()
        # Define Slow Pathway (slow-moving features)
        self.slow_path = SlowPathway(input_channels=3, base_channels=slow_channels)

        # Define Fast Pathway (fast-moving features)
        self.fast_path = FastPathway(input_channels=3, base_channels=fast_channels)

        # Lateral Connections for feature integration
        self.lateral_connections = LateralConnections(
            slow_channels=slow_channels * 4, fast_channels=fast_channels * 4
        )

        # Final fully connected layer for classification
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(slow_channels * 4 + fast_channels * 4, num_classes)
        )

    def forward(self, x_slow, x_fast):
        """
        Forward pass of the SlowFast model.
        Args:
            x_slow: Input tensor for the Slow Pathway.
            x_fast: Input tensor for the Fast Pathway.
        Returns:
            logits: Predicted class logits.
        """
        # Pass inputs through slow and fast pathways
        slow_features = self.slow_path(x_slow)
        fast_features = self.fast_path(x_fast)

        # Integrate features using lateral connections
        combined_features = self.lateral_connections(slow_features, fast_features)

        # Classify with fully connected layer
        logits = self.fc(combined_features)
        return logits

if __name__ == "__main__":
    # Testing the model
    model = SlowFastModel(num_classes=10)
    x_slow = torch.randn(2, 3, 32, 224, 224)  # Batch of 2, slow pathway input
    x_fast = torch.randn(2, 3, 8, 224, 224)   # Batch of 2, fast pathway input
    logits = model(x_slow, x_fast)
    print("Output logits shape:", logits.shape)  # Should be [2, num_classes]

import torch
from sklearn.metrics import accuracy_score

class EvalUtils:
    """
    Utility class for evaluating SlowFast models.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy
        acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {acc:.4f}")
        return acc

import torch
import torch.nn.functional as F


class DiceCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, dice_weight=1.0, ignore_index=255):
        super(DiceCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='mean' if self.size_average else 'sum',
                                  ignore_index=self.ignore_index)

        # Ensure inputs and targets have the same shape for Dice Loss
        mask = (targets != self.ignore_index)

        inputs = torch.sigmoid(inputs)
        inputs = inputs * mask.unsqueeze(1)  # Apply mask
        targets = targets * mask

        # Flatten inputs and targets while ignoring masked-out elements
        inputs_flat = inputs[:, 1:, :, :].contiguous().view(-1)  # Use class channel if inputs are logits
        targets_flat = targets.view(-1)

        # Dice Loss
        smooth = 1e-5
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth))

        # Combined Loss
        loss = ce_loss + self.dice_weight * dice_loss
        return loss

# Usage Example
# model = ...
# criterion = DiceCrossEntropyLoss(ignore_index=255)
# optimizer = ...

# for inputs, targets in dataloader:
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     loss.backward()
#     optimizer.step()

# bpc/pose/models/simple_pose_net.py

import torch
import torch.nn as nn
import torchvision.models as tv_models

class SimplePoseNet(nn.Module):
    def __init__(self, loss_type="euler", pretrained=True):
        """
        Args:
            loss_type: one of "euler", "quat", or "6d". Determines the number of output neurons.
            pretrained: if True, use pretrained ResNet50 weights.
        """
        super(SimplePoseNet, self).__init__()
        # Load a ResNet50 backbone.
        backbone = tv_models.resnet50(
            weights=(tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        )
        layers = list(backbone.children())[:-1]  # Remove the classification head.
        self.backbone = nn.Sequential(*layers)
        
        # Determine rotation output dimension.
        if loss_type == "euler":
            out_dim = 3
        elif loss_type == "quat":
            out_dim = 4
        elif loss_type == "6d":
            out_dim = 6
        else:
            raise ValueError("loss_type must be one of 'euler', 'quat', or '6d'")
        
        self.fc = nn.Linear(2048, out_dim)  # Only rotation outputs.

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        preds = self.fc(feats)
        return preds

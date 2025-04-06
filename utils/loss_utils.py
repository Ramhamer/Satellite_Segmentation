import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DiceCrossEntropyLoss(nn.Module):
    """
    Dice + Cross Entropy:
    Dice Loss: Helps with class imbalance by focusing on the overlap between predicted and ground truth masks.
    Cross Entropy Loss: Works well for multi-class segmentation, ensuring the network correctly classifies each pixel.

    Dice compensates for class imbalance, making sure small objects aren’t ignored.
    Cross Entropy stabilizes training, ensuring the model learns pixel-wise classifications correctly.
    """    
    
    def __init__(self, loss_mode="multiclass", weight=None, dice_weight=0.5, ce_weight=0.5, log_loss=False, from_logits=True, smooth=1e-6, ignore_index=255, eps=1e-7):
        """
        Combined Dice Loss + Cross Entropy Loss.

        Args:
            loss_mode (str): "binary", "multiclass", or "multilabel".
            weight (tensor): Class weights for Cross Entropy Loss.
            dice_weight (float): Weight of Dice Loss in total loss.
            ce_weight (float): Weight of Cross Entropy Loss in total loss.
            log_loss (bool): If True, use log loss for Dice.
            from_logits (bool): If True, model outputs raw logits.
            smooth (float): Smoothing factor for Dice Loss.
            ignore_index (int): Index to ignore in loss calculation.
            eps (float): Small value to prevent division by zero.
        """
        super(DiceCrossEntropyLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode=loss_mode, log_loss=log_loss, from_logits=from_logits, smooth=smooth, ignore_index=ignore_index, eps=eps)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, preds, targets):
        """
        Compute the combined loss.
        Args:
            preds (tensor): Model output (logits).
            targets (tensor): Ground truth labels.
        Returns:
            Tensor: Total loss value.
        """
        dice = self.dice_loss(preds, targets)
        ce = self.ce_loss(preds, targets)
        return self.dice_weight * dice + self.ce_weight * ce


class JaccardFocalLoss(nn.Module):
    """
   Jaccard + Focal:
    Jaccard Loss (IoU Loss): Similar to Dice, it measures the overlap but penalizes false positives more aggressively.
    Focal Loss: Focuses on hard-to-classify pixels, especially useful when some classes are rare.

    Jaccard helps with spatial accuracy, ensuring segmented areas align with real objects.
    Focal prevents the model from ignoring small/rare classes, boosting performance on minority regions.
    """ 

    def __init__(self, loss_mode="multiclass", weight=None, jaccard_weight=0.5, focal_weight=0.5, ignore_index=255, eps=1e-7):
        """
        Combined Jaccard Loss (IoU) + Focal Loss.

        Args:
            loss_mode (str): "binary", "multiclass", or "multilabel".
            weight (tensor): Class weights for Focal Loss.
            jaccard_weight (float): Weight of Jaccard Loss in total loss.
            focal_weight (float): Weight of Focal Loss in total loss.
            ignore_index (int): Index to ignore in loss calculation.
            eps (float): Small value to prevent division by zero.
        """
        super(JaccardFocalLoss, self).__init__()
        self.jaccard_loss = smp.losses.JaccardLoss(mode=loss_mode, from_logits=True, eps=eps)
        self.focal_loss = smp.losses.FocalLoss(mode=loss_mode, ignore_index=ignore_index)
        self.jaccard_weight = jaccard_weight
        self.focal_weight = focal_weight

    def forward(self, preds, targets):
        """
        Compute the combined loss.
        Args:
            preds (tensor): Model output (logits).
            targets (tensor): Ground truth labels.
        Returns:
            Tensor: Total loss value.
        """
        jaccard = self.jaccard_loss(preds, targets)
        focal = self.focal_loss(preds, targets)
        return self.jaccard_weight * jaccard + self.focal_weight * focal

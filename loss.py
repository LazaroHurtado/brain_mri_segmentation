import torch

class Loss:

    def __init__(self, criterion: torch.nn.Module, gpu_id: int):
        self.criterion = criterion
        self.device = gpu_id
    
    def dice_coeff_fn(self, y_pred: torch.Tensor, y: torch.Tensor, smooth: float = 1e-6):
        y_pred = (y_pred > 0.5).float()
        y = (y > 0.5).float()
        
        intersection = 2 * (y_pred * y).sum(dim=(1,2,3)) + smooth
        union = (y_pred + y).sum(dim=(1,2,3)) + smooth

        return (intersection / union).mean()

    def dice_loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor, smooth: float = 1e-6):
        return 1 - self.dice_coeff_fn(y_pred, y, smooth)

    def focal_loss_fn(self,
                      y_pred: torch.Tensor,
                      y: torch.Tensor,
                      weights: torch.Tensor,
                      gamma: float = 2.0):
        bce_loss = weights * self.criterion(y_pred, y)
        probs = y * y_pred + (1.0 - y) * (1.0 - y_pred)

        loss = (1 - probs).pow(gamma) * bce_loss
        return loss

    def __call__(self,
                 y_pred: torch.Tensor,
                 y: torch.Tensor,
                 weights: torch.Tensor,
                 betas: tuple = (5.0, 1.0)):
        beta1, beta2 = betas
        weights = weights.reshape(-1, 1, 1, 1).reciprocal().to(self.device)

        dice_loss = self.dice_loss_fn(y_pred, y)
        focal_loss = self.focal_loss_fn(y_pred, y, weights, gamma=2.0)

        loss = beta1*focal_loss + beta2*dice_loss
        return loss.mean()
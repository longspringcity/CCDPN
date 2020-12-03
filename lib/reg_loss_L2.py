import torch


class RegLoss(torch.nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, pred_pos, real_pos):
        error_distances = torch.norm(pred_pos - real_pos, 2, dim=1)
        total_loss = torch.sum(error_distances)
        return total_loss

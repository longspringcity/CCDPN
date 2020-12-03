import torch


class RegLoss(torch.nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, pred_pos, real_pos):
        diff_loss = abs(pred_pos - real_pos)
        total_loss = torch.sum(diff_loss)
        return total_loss

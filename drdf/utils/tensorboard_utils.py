from re import L

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, log_dir):
        super().__init__(log_dir)

    def log_gradients_norms(self, model, step):
        for tag, value in model.named_parameters():
            if value.grad is not None:
                grad_norm = value.grad.norm(2).item()
                self.add_scalar(tag + "/grad", grad_norm, step)

    def log_model_grad_norm(self, model, step):
        total_norm = 0.0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.add_scalar("/grad_norm", total_norm, step)

import torch


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True,
    )[0]
    return grad

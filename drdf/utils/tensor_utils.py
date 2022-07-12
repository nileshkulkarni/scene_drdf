import numpy as np
import torch


def tensor_to_cuda(tensor, cuda, tensor_type=torch.FloatTensor):
    if not (torch.Tensor == type(tensor)):
        # print('coming here')
        tensor = tensor_type(tensor * 1)
    return tensor.cuda() if cuda else tensor.cpu()
    if cuda == True:
        tensor = tensor.cuda()
    else:
        tensor = tensor.cpu()
    return tensor


def tensor_to_numpy(tensor):
    if type(tensor) == torch.Tensor:
        if tensor.device.type == "cuda":
            tensor = tensor.cpu()
        return tensor.data.numpy()
    elif type(tensor) == np.ndarray:
        return tensor
    else:
        return tensor

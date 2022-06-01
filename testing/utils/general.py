import torch


def set_default_device_cuda(dont_use_cuda=False):
    """Sets the default device (cpu or cuda) used for all tensors."""
    if torch.cuda.is_available() == False or dont_use_cuda:
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return False
    else:  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return True
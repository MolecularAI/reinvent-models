# coding=utf-8

"""
Generic utilities.
"""
import torch


def dynamic_tensor_allocation(tensor):
    """
    Dynamically allocation of a tensor based on GPU availability.
    :param tensor: A torch tensor.
    :return: The tensor properly allocated.
    """
    tensor.cuda() if torch.cuda.is_available() else tensor


def load_with_dynamic_map_location(path):
    """
    Dynamically load an artifact based on GPU availability.
    :param str: path to the artifact to load.
    :return: The loaded artifact with the appropriate map_location.
    """
    torch.load(path) if torch.cuda.is_available() else torch.load(path, map_location=torch.device('cpu'))

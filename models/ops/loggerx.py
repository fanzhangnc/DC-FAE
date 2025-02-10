# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : loggerx.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 8:47 PM 
'''
import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect

from . import load_network


def get_varname(var):
    """
    Gets the name of a variable. Searches from the outermost frame inward.

    Parameters:
    var (any): The variable to get the name of.

    Returns:
    str: The name of the variable as a string.
    """
    for fi in reversed(inspect.stack()):
        # List comprehension to find variable names in the frame's local variables
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            # Return the first variable name found
            return names[0]


def reduce_tensor(rt):
    rt = rt.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    rt /= world_size
    return rt


class LoggerX:
    """
    LoggerX class for saving models and images, with optional Weights and Biases (wandb) logging.
    """

    def __init__(self, save_root, enable_wandb=False, **kwargs):
        """
        Initialize LoggerX.

        Parameters:
        save_root (str): Root directory for saving models and images.
        enable_wandb (bool): Enable wandb logging.
        kwargs: Additional arguments for wandb.init.
        """
        if save_root is not None:
            # Set directories for saving models and images
            self.models_save_dir = osp.join(save_root, 'save_models')
            self.images_save_dir = osp.join(save_root, 'save_images')
            # Create directories if they do not exist
            os.makedirs(self.models_save_dir, exist_ok=True)
            os.makedirs(self.images_save_dir, exist_ok=True)

        self._modules = []  # List to store module references
        self._module_names = []  # List to store module names
        self.local_rank = 0  # Local rank, assuming single process
        self.enable_wandb = enable_wandb  # Flag to enable wandb logging

        if enable_wandb and self.local_rank == 0:
            import wandb  # Import wandb if logging is enabled
            # Initialize wandb with the given settings and arguments
            wandb.init(
                dir=save_root,
                settings=wandb.Settings(
                    _disable_stats=True,
                    _disable_meta=True
                ),
                **kwargs
            )

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for module in modules:
            self._modules.append(module)
            self._module_names.append(get_varname(module))

    def append(self, module, name=None):
        self._modules.append(module)
        if name is None:
            name = get_varname(module)
        self._module_names.append(name)

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        """
        Load checkpoints for all registered modules.

        Parameters:
        epoch (int): The epoch number used to identify the checkpoint files.
        """
        for module_name, module in zip(self.module_names, self.modules):
            checkpoint_path = osp.join(self.models_save_dir, f'{module_name}-{epoch}')
            module.load_state_dict(load_network(checkpoint_path))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)
        output_dict = {}
        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)
            output_dict[var_name] = var

        if self.enable_wandb and self.local_rank == 0:
            import wandb
            wandb.log(output_dict, step)

        if self.local_rank == 0:
            print(output_str)

    def msg_str(self, output_str):
        if self.local_rank == 0:
            print(str(output_str))

    def save_image(self, grid_img, n_iter, sample_type):
        """
        Save the image to the specified directory.

        Args:
            grid_img (Image): The PIL Image to be saved.
            n_iter (int): The current iteration number.
            sample_type (str): The type of sample (e.g., 'train', 'test').
        """
        filename = f"{n_iter}_{self.local_rank}_{sample_type}.jpg"
        file_path = osp.join(self.images_save_dir, filename)
        grid_img.save(file_path)
